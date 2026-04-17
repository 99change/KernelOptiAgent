/*
 * latency_hiding.cu
 * CUDA 内存延迟隐藏示例：软件流水线（register double-buffering）+ __ldg 只读缓存
 *
 * 关键禁忌：
 *   __builtin_prefetch() 是 GCC host 函数，不能在 __global__ / __device__ 函数中使用！
 *   在 CUDA kernel 里调用会得到编译错误：
 *     "calling a __host__ function from a __global__ function is not allowed"
 *
 * CUDA 中延迟隐藏的正确手段：
 *   1. __ldg(&ptr[i])        — 通过只读 texture cache 加载，减少 L2 压力
 *   2. register 双缓冲       — 提前把下一迭代数据 load 到寄存器，当前迭代计算期间等 load 完成
 *   3. ILP（指令级并行）     — 同一线程内展开多次独立 load，让硬件调度器重叠执行
 *   4. cuda::memcpy_async    — cp.async 硬件异步拷贝（sm_80+），配合 __syncthreads_async 使用
 *
 * 本文件演示方法 2（register 双缓冲）和方法 3（ILP）。
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 20)

// ─────────────────────────────────────────────
// 方法 1：__ldg 只读缓存（最简单，推荐首选）
// ─────────────────────────────────────────────
__global__ void vector_add_ldg(const float * __restrict__ a,
                                const float * __restrict__ b,
                                float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // __ldg 通过只读 L1 cache 加载，减少 bandwidth 压力
        c[i] = __ldg(&a[i]) + __ldg(&b[i]);
    }
}

// ─────────────────────────────────────────────
// 方法 2：ILP 展开（每线程处理 4 个元素，提高 MLP）
// 4 个独立 load 可被硬件重叠执行，有效隐藏单次 load 延迟
// ─────────────────────────────────────────────
__global__ void vector_add_ilp4(const float * __restrict__ a,
                                 const float * __restrict__ b,
                                 float *c, int n) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // 4 个独立 load 放在一起，允许硬件乱序执行（MLP = memory-level parallelism）
    float a0 = (base + 0 < n) ? __ldg(&a[base + 0]) : 0.f;
    float a1 = (base + 1 < n) ? __ldg(&a[base + 1]) : 0.f;
    float a2 = (base + 2 < n) ? __ldg(&a[base + 2]) : 0.f;
    float a3 = (base + 3 < n) ? __ldg(&a[base + 3]) : 0.f;

    float b0 = (base + 0 < n) ? __ldg(&b[base + 0]) : 0.f;
    float b1 = (base + 1 < n) ? __ldg(&b[base + 1]) : 0.f;
    float b2 = (base + 2 < n) ? __ldg(&b[base + 2]) : 0.f;
    float b3 = (base + 3 < n) ? __ldg(&b[base + 3]) : 0.f;

    // 计算（此时 load 大概率已完成）
    if (base + 0 < n) c[base + 0] = a0 + b0;
    if (base + 1 < n) c[base + 1] = a1 + b1;
    if (base + 2 < n) c[base + 2] = a2 + b2;
    if (base + 3 < n) c[base + 3] = a3 + b3;
}

// ─────────────────────────────────────────────
// 方法 3：register 双缓冲软件流水线（适合有 loop 的 kernel）
// 典型场景：reduce、scan、stencil 等需要多轮全局访存的 kernel
// ─────────────────────────────────────────────
__global__ void reduce_pipelined(const float * __restrict__ in,
                                  float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.f;

    // 预取第一个元素（软件流水第 0 步）
    float prefetched = (i < n) ? __ldg(&in[i]) : 0.f;
    i += stride;

    // 流水主循环：计算当前值的同时发出下一次 load
    while (i < n) {
        float current = prefetched;           // 使用已到达的数据
        prefetched = __ldg(&in[i]);           // 发出下一次 load（异步进行）
        sum += current;                        // 计算与 load 重叠
        i += stride;
    }
    sum += prefetched;                         // 处理最后一个

    // 简单归约写回（实际使用时接 shared memory reduce）
    atomicAdd(out, sum);
}

int main() {
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) { h_a[i] = 1.f; h_b[i] = 2.f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // ILP4 版本：每线程处理 4 元素，grid 缩小 4 倍
    int blockSize = 256;
    int gridSize = (N / 4 + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    vector_add_ilp4<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("EXEC_TIME_MS:%.4f\n", ms);

    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        if (h_c[i] != 3.f) { printf("MISMATCH at %d\n", i); return 1; }
    }
    printf("PASSED\n");

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
