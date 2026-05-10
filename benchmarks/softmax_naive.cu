/*
 * softmax_naive.cu
 * Naive row-wise softmax on a 2-D matrix (B × D).
 * KernelBench Level 1 representative: reduction operator.
 *
 * Optimization opportunities:
 * 1. Three separate passes (max / exp / sum / divide) — poor temporal locality
 * 2. No shared memory: all accesses go to global memory
 * 3. One warp/block per row — low occupancy for large D
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NROWS  4096
#define NCOLS  1024

__global__ void softmax_naive(const float *__restrict__ x, float *y, int nrows, int ncols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    const float *row_x = x + row * ncols;
    float       *row_y = y + row * ncols;

    /* pass 1: max */
    float mx = row_x[0];
    for (int j = 1; j < ncols; j++) mx = fmaxf(mx, row_x[j]);

    /* pass 2: sum of exp */
    float sum = 0.0f;
    for (int j = 0; j < ncols; j++) sum += expf(row_x[j] - mx);

    /* pass 3: normalise */
    for (int j = 0; j < ncols; j++) row_y[j] = expf(row_x[j] - mx) / sum;
}

int main() {
    size_t bytes = (size_t)NROWS * NCOLS * sizeof(float);
    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    for (int i = 0; i < NROWS * NCOLS; i++) h_x[i] = (float)rand() / RAND_MAX;

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

    /* one thread per row */
    int threads = 256;
    int blocks  = (NROWS + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    softmax_naive<<<blocks, threads>>>(d_x, d_y, NROWS, NCOLS);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    softmax_naive<<<blocks, threads>>>(d_x, d_y, NROWS, NCOLS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("time: %.3f ms\n", ms);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    free(h_x); free(h_y);
    cudaFree(d_x); cudaFree(d_y);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
