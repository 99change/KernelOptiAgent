/*
 * transpose_naive.cu
 * Naive square matrix transpose: out[j][i] = in[i][j].
 * KernelBench Level 1 representative: memory-access-pattern operator.
 *
 * Optimization opportunities:
 * 1. Un-coalesced global memory writes (strided column access)
 * 2. No shared-memory tiling to improve write coalescing
 * 3. No vectorized loads
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4096   /* N × N matrix */

__global__ void transpose_naive(const float *__restrict__ in, float *out, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        out[col * n + row] = in[row * n + col];
    }
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 threads(32, 8);
    dim3 blocks((N + 31) / 32, (N + 7) / 8);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    transpose_naive<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    transpose_naive<<<blocks, threads>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("time: %.3f ms\n", ms);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
