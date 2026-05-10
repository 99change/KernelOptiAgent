/*
 * relu_naive.cu
 * Naive elementwise ReLU on a large 1-D tensor.
 * KernelBench Level 1 representative: elementwise unary operator.
 *
 * Optimization opportunities:
 * 1. No vectorized loads (float4 possible)
 * 2. Single element per thread — low arithmetic intensity
 * 3. No __ldg() cache hint on read
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 25)   /* 33.5 M elements, ~128 MB at float32 */

__global__ void relu_naive(const float *__restrict__ input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

int main() {
    float *h_in, *h_out;
    float *d_in, *d_out;
    size_t bytes = (size_t)N * sizeof(float);

    h_in  = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 200 - 100) / 10.0f;

    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* warm-up */
    relu_naive<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    relu_naive<<<blocks, threads>>>(d_in, d_out, N);
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
