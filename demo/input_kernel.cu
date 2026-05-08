/*
 * matmul_naive.cu
 * 一个完全未优化的矩阵乘法，有更多的优化空间（比 vector_add 更有挑战性）。
 *
 * 主要问题：
 * 1. 没有 shared memory tiling（大量重复的 global memory 访问）
 * 2. 内存访问 stride 不是最优的
 * 3. 没有向量化读取
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define M 4096
#define K 4096
#define N 4096

// 最朴素的矩阵乘法 kernel
__global__ void matmul_naive(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10) / 10.0f;

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %.3f ms\n", milliseconds);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
