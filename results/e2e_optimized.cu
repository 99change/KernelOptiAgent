#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define M 4096
#define K 4096
#define N 4096

// Optimized Matrix Multiplication Kernel using Shared Memory Tiling
__global__ void matmul_naive(float *A, float *B, float *C, int m, int k, int n) {
    // Define tile dimensions
    #define TILE_WIDTH 16
    
    // Shared memory declarations with padding for Bs to avoid bank conflicts
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH + 1];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (row < m && col < n) {
        float psum = 0.0f;

        // Loop over tiles of A and B
        for (int t = 0; t < ((k + TILE_WIDTH - 1) / TILE_WIDTH); ++t) {
            // Load A tile into shared memory
            // Coalesced access: threadIdx.x varies along row of A
            int a_val = (t * TILE_WIDTH + threadIdx.x < k) ? 
                        A[row * k + t * TILE_WIDTH + threadIdx.x] : 0.0f;
            As[threadIdx.y][threadIdx.x] = a_val;

            // Load B tile into shared memory
            // Coalesced access: threadIdx.x varies along row of B (column-major in logic, row-major in memory)
            // We load B such that Bs[ty][tx] corresponds to B[k_start + ty][col + tx]
            int b_val = ((t * TILE_WIDTH + threadIdx.y < k) && (col + threadIdx.x < n)) ? 
                        B[(t * TILE_WIDTH + threadIdx.y) * n + col + threadIdx.x] : 0.0f;
            Bs[threadIdx.y][threadIdx.x] = b_val;

            __syncthreads();

            // Perform multiplication for the current tile
            for (int i = 0; i < TILE_WIDTH; ++i) {
                psum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
            }

            __syncthreads();
        }

        C[row * n + col] = psum;
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

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

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