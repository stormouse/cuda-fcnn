#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "functions.cuh"
#include "nn.cuh"

#include <vector>
#include <random>

std::vector<float> generateRandomVector(size_t size) {
    std::vector<float> v(size);

    // Random number generator
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator

    // Define the range for uniform distribution
    std::uniform_real_distribution<> dis(0.0, 1.0);  // Range [0.0, 1.0)

    // Populate the vector with random numbers
    for (auto& elem : v) {
        elem = dis(gen) >= 0.5f ? 1.0f : 0.0f;  // Each call to dis(gen) generates a new random float
    }

    return v;
}

int main() {
    float *d_input;
    int dim = 16;
    CUDA_CALL(cudaMalloc((void **)&d_input, dim * sizeof(float)));

    std::vector<int> layers{dim, 4, dim};
    Model model(layers);
    float lr = 0.01f;

    auto h_input = generateRandomVector(dim);

    for (int t = 0; t < 100; t++)
    {
        float epochLoss = 0.0f;
        for (int i = 0; i < 1000; i++)
        {
            h_input = generateRandomVector(dim);
            CUDA_CALL(cudaMemcpy(d_input, h_input.data(), dim * sizeof(float), cudaMemcpyHostToDevice));

            auto loss = model.fit(d_input, d_input, lr);
            epochLoss += loss;
        }
        std::cout << "epoch " << t << ": loss = " << epochLoss / 1000 << ", lr = " << lr << "\n";
        lr *= 0.98f;
    }

    for (int i = 0; i < layers.size() - 1; i++) {
        std::cout << "W:" << std::endl;
        printKernel<<<1, 1>>>(model.layers[i]->W, layers[i] * layers[i + 1]);
        cudaDeviceSynchronize();
        std::cout << "b:" << std::endl;
        printKernel<<<1, 1>>>(model.layers[i]->b, layers[i + 1]);
        cudaDeviceSynchronize();
        std::cout << "a:" << std::endl;
        printKernel<<<1, 1>>>(model.layers[i]->a, layers[i + 1]);
        cudaDeviceSynchronize();
        std::cout << "\n\n\n" << std::endl;
    }

    printKernel<<<1, 1>>>(model.layers.back()->a, layers.back());
    cudaDeviceSynchronize();
    for (int i = 0; i < layers.back(); i++) {
        std::cout << "Label[" << i << "] = " << h_input[i] << "\n";
    }

    std::cout << "\n";

    cudaDeviceSynchronize();

    CUDA_CALL(cudaFree(d_input));

    return 0;
}