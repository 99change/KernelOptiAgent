/*
 * ================================================================
 *  KernelOptiAgent - Optimization Summary
 *  Generated : 2026-05-08 14:40:57
 * ================================================================
 *
 *  Baseline time  : 91.871 ms
 *  Optimized time : 62.000 ms
 *  Total speedup  : 32.5%
 *
 *  Bottlenecks identified:
 *    - shared_memory_underused (score=0.97, evidence: smem_bytes=0, data_reuse_possible=True)
 *    - non_coalesced_memory (score=0.70, evidence: access_pattern=strided)
 *
 *  Changes applied:
 *    [1] Tile global memory accesses through shared memory to exploit data reuse
 *    [2] Coalesce memory access so consecutive threads access consecutive addresses
 *
 * ================================================================
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define M 4096
#define K 4096
#define N 4096
#define TILE_SIZE 16

// Optimized matrix multiplication kernel using shared memory tiling
__global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int m, int k, int n)
{
    // Declare shared memory tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Iterate over tiles along the K dimension
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Each thread cooperates to load elements into shared memory
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? A[row * k + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < k && col < n) ? B[bRow * n + col] : 0.0f;

        // Synchronize to ensure all threads have finished loading
        __syncthreads();

        // Perform dot product for the current tile
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        // Synchronize before loading the next tile to prevent overwriting
        __syncthreads();
    }

    if (row < m && col < n) {
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

    // Initialize host data
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10) / 10.0f;

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

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