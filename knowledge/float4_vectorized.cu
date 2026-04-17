/*
 * float4_vectorized.cu
 * 向量化内存访问示例：使用 float4 一次加载/存储 4 个 float
 *
 * 关键点：
 * 1. float4 没有重载的 + 运算符，必须对 .x .y .z .w 分量分别操作
 * 2. N 必须是 4 的倍数，或者尾部单独处理
 * 3. 指针需要转型为 float4*
 * 4. __ldg() 接受指针（const T*），不接受值（T）！必须传 &ptr[idx] 或 ptr+idx
 *    错误: __ldg(reinterpret_cast<float4*>(a)[idx])   // 传了 float4 值，编译报错
 *    正确: __ldg(&reinterpret_cast<const float4*>(a)[idx])  // 传指针，OK
 *    正确: __ldg(reinterpret_cast<const float4*>(a) + idx)  // 传指针，OK
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 20)

// 正确的 float4 向量加法 kernel（无 __ldg）
__global__ void vector_add_float4(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;

    if (idx < n4) {
        // 关键：转型为 float4* 再读取，一次读 16 bytes
        float4 va = reinterpret_cast<float4*>(a)[idx];
        float4 vb = reinterpret_cast<float4*>(b)[idx];
        float4 vc;

        // 注意：CUDA C 的 float4 不支持直接 + 运算符
        // 必须对每个分量分别操作
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;

        reinterpret_cast<float4*>(c)[idx] = vc;
    }

    // 处理不能被 4 整除的尾部
    int tail_start = (n / 4) * 4;
    if (idx == 0) {
        for (int i = tail_start; i < n; i++) {
            c[i] = a[i] + b[i];
        }
    }
}

// float4 + __ldg() 正确用法示例
// __ldg() 必须接受指针 (const T*)，不接受值 (T)！
// 错误（编译报错）:  __ldg(reinterpret_cast<float4*>(b)[idx])
// 正确:             __ldg(&reinterpret_cast<const float4*>(b)[idx])
// 正确(等价写法):    __ldg(reinterpret_cast<const float4*>(b) + idx)
__global__ void vector_add_float4_ldg(const float * __restrict__ a,
                                       const float * __restrict__ b,
                                       float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;

    if (idx < n4) {
        // 正确：传指针给 __ldg，返回 float4 值
        float4 va = __ldg(&reinterpret_cast<const float4*>(a)[idx]);
        float4 vb = __ldg(&reinterpret_cast<const float4*>(b)[idx]);
        float4 vc;

        // float4 必须逐分量操作
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;

        reinterpret_cast<float4*>(c)[idx] = vc;
    }
}

int main() {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // grid/block 按 N/4 配置（每个线程处理 4 个元素）
    int blockSize = 256;
    int gridSize = (N / 4 + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vector_add_float4<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("EXEC_TIME_MS:%.4f\n", ms);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 验证
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("MISMATCH at %d\n", i);
            return 1;
        }
    }
    printf("PASSED\n");

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
