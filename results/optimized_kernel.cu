/*
 * ================================================================
 *  KernelOptiAgent - Optimization Summary
 *  Generated : 2026-05-07 23:18:18
 * ================================================================
 *
 *  Baseline time  : 0.563 ms
 *  Optimized time : 0.433 ms
 *  Total speedup  : 23.0%
 *
 *  Bottlenecks identified:
 *    - memory_bound (score=1.00, evidence: memory_throughput_pct=70.2, dram_throughput_pct=35.5)
 *    - compute_underutilized (score=1.00, evidence: compute_throughput_pct=24.3)
 *    - shared_memory_underused (score=1.00, evidence: smem_bytes=0, data_reuse_possible=True)
 *
 *  Changes applied:
 *    [1] Use float4 vectorized loads and __ldg() to increase memory throughput
 *    [2] Increase arithmetic intensity by loop unrolling (#pragma unroll), ILP (each thread handles multiple elements), or fusing adjacent element-wise operations into one kernel. Do NOT use tensor cores unless the kernel already performs matrix multiply.
 *
 * ================================================================
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 20)  // 1M elements

// Optimized vector addition kernel using float4 and __ldg
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;

    if (idx < n4) {
        // Use __ldg to load data with float4
        float4 va = __ldg(reinterpret_cast<const float4*>(a) + idx);
        float4 vb = __ldg(reinterpret_cast<const float4*>(b) + idx);
        float4 vc;

        // Perform the addition on each component of the float4
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;

        // Store the result back as float4
        reinterpret_cast<float4*>(c)[idx] = vc;
    }

    // Handle the tail that is not a multiple of 4
    int tail_start = (n / 4) * 4;
    if (idx == 0) {
        for (int i = tail_start; i < n; i++) {
            c[i] = a[i] + b[i];
        }
    }
}

int main() {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Configure kernel execution parameters
    int blockSize = 256;
    int gridSize = (N / 4 + blockSize - 1) / blockSize;

    // Timing
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

    // Copy results back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify results
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            correct = false;
            break;
        }
    }
    printf("Result: %s\n", correct ? "CORRECT" : "WRONG");

    // Free memory
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}