#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "nn.cuh"

int main() {
    float *h_A, *h_B, *h_C; // Host pointers
    float *d_A, *d_B, *d_C; // Device pointers
    float alpha = 1.0f;
    float beta = 0.5f;

    // Allocate host memory
    h_A = new float[16 * 64];
    h_B = new float[64 * 1];
    h_C = new float[16 * 1];

    // Initialize matrices on the host
    for (int i = 0; i < 16 * 64; ++i) {
        h_A[i] = (i % 4) * 1.0f;
        if (i < 64) h_B[i] = 4.0f / (i + 1.0f);
        h_C[i % 16] = 0.3f;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_A, 16 * 64 * sizeof(float));
    cudaMalloc((void **)&d_B, 64 * 1 * sizeof(float));
    cudaMalloc((void **)&d_C, 16 * 1 * sizeof(float));

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A, 16 * 64 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 64 * 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, 16 * 1 * sizeof(float), cudaMemcpyHostToDevice);

    // Create a handle for cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    cublasSgemm(handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      16, 1, 64,
      &alpha,
      d_A, 16,
      d_B, 64,
      &beta,
      d_C, 16);

    // float one = 1.0f;
    // cublasSgeam(handle,
    //     CUBLAS_OP_N,
    //     CUBLAS_OP_N,
    //     16, 1,
    //     &one,
    //     d_C, 16,
    //     &one,
    //     d_C, 16,
    //     d_C, 16);

    // Copy the result back to host
    cudaMemcpy(h_C, d_C, 16 * 1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Example: Print the first 10 elements of the result matrix
    for (int i = 0; i < 16; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    // Destroy the handle
    cublasDestroy(handle);

    return 0;
}