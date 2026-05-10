/*
 * layernorm_naive.cu
 * Naive layer normalization over the last dimension (B × D).
 * KernelBench Level 1 representative: compound reduction + elementwise.
 *
 * Optimization opportunities:
 * 1. Two global-memory passes for mean & variance
 * 2. No shared-memory reductions — each pass is fully serial per row
 * 3. No warp-shuffle reductions
 * 4. Division by N instead of rsqrtf fusion
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NROWS 4096
#define NCOLS 1024
#define EPS  1e-5f

__global__ void layernorm_naive(const float *__restrict__ x,
                                 const float *__restrict__ gamma,
                                 const float *__restrict__ beta,
                                 float *y, int nrows, int ncols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    const float *row_x = x + row * ncols;
    float       *row_y = y + row * ncols;

    /* mean */
    float mean = 0.0f;
    for (int j = 0; j < ncols; j++) mean += row_x[j];
    mean /= ncols;

    /* variance */
    float var = 0.0f;
    for (int j = 0; j < ncols; j++) {
        float diff = row_x[j] - mean;
        var += diff * diff;
    }
    var /= ncols;
    float inv_std = 1.0f / sqrtf(var + EPS);

    /* normalise + affine */
    for (int j = 0; j < ncols; j++)
        row_y[j] = gamma[j] * (row_x[j] - mean) * inv_std + beta[j];
}

int main() {
    size_t bytes   = (size_t)NROWS * NCOLS * sizeof(float);
    size_t bytes_d = (size_t)NCOLS * sizeof(float);

    float *h_x     = (float*)malloc(bytes);
    float *h_y     = (float*)malloc(bytes);
    float *h_gamma = (float*)malloc(bytes_d);
    float *h_beta  = (float*)malloc(bytes_d);

    for (int i = 0; i < NROWS * NCOLS; i++) h_x[i] = (float)rand() / RAND_MAX;
    for (int j = 0; j < NCOLS; j++) { h_gamma[j] = 1.0f; h_beta[j] = 0.0f; }

    float *d_x, *d_y, *d_gamma, *d_beta;
    cudaMalloc(&d_x,     bytes);
    cudaMalloc(&d_y,     bytes);
    cudaMalloc(&d_gamma, bytes_d);
    cudaMalloc(&d_beta,  bytes_d);

    cudaMemcpy(d_x,     h_x,     bytes,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, bytes_d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,  h_beta,  bytes_d, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (NROWS + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    layernorm_naive<<<blocks, threads>>>(d_x, d_gamma, d_beta, d_y, NROWS, NCOLS);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    layernorm_naive<<<blocks, threads>>>(d_x, d_gamma, d_beta, d_y, NROWS, NCOLS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("time: %.3f ms\n", ms);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    free(h_x); free(h_y); free(h_gamma); free(h_beta);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma); cudaFree(d_beta);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
