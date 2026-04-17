/*
 * ================================================================
 *  KernelOptiAgent - Optimization Summary
 *  Generated : 2026-04-17 17:27:03
 * ================================================================
 *
 *  Baseline time  : 0.345 ms
 *  Optimized time : 0.345 ms
 *  Total speedup  : 0.0%
 *
 *  Bottlenecks identified:
 *    - memory_bound (score=1.00, evidence: arithmetic_intensity=low, loads_per_flop=2.0)
 *    - memory_latency_bound (score=0.90, evidence: independent_loads=True)
 *    - compute_underutilized (score=0.67, evidence: flops_per_element=1.0)
 *
 *  Changes applied:
 *    [1] Use float4 vectorized loads and __ldg() to increase memory throughput
 *
 * ================================================================
 */
/*
 * vector_add.cu
 * 一个未优化的向量加法 kernel，用于测试优化 Agent。
 *
 * 主要问题（Agent 应该能发现）：
 * 1. 没有利用 shared memory
 * 2. 内存访问可能不是最优的
 * 3. 没有使用任何并行优化技巧
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 20)  // 1M elements

// 简单的向量加法 kernel（未优化）
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // 分配主机内存
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // 分配设备内存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 拷贝数据到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 配置 kernel 执行参数
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %.3f ms\n", milliseconds);

    // 拷贝结果回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 验证结果
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            correct = false;
            break;
        }
    }
    printf("Result: %s\n", correct ? "CORRECT" : "WRONG");

    // 释放内存
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
