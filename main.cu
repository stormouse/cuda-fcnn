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

int main()
{
    int dim = 16;
    int numInputs = 100;
    auto inputs = std::vector<float*>(numInputs, nullptr);
    
    for (int i = 0; i < numInputs; i++) {
        auto h_input = generateRandomVector(dim);
        CUDA_CALL(cudaMalloc((void **)&inputs[i], dim * sizeof(float)));
        CUDA_CALL(cudaMemcpy(inputs[i], h_input.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }

    // there is a bug - {dim, dim-1, dim} cannot learn but {dim, dim, dim} can learn perfectly well
    // something about the matrix shape must be wrong, causing buffer overflow (and square matrix mitigated that)
    std::vector<int> layers{dim, dim, dim};
    Model model(layers);
    float lr = 0.1f;

    for (int t = 0; t < 100; t++)
    {
        float epochLoss = 0.0f;
        for (int i = 0; i < inputs.size(); i++)
        {
            auto loss = model.fit(inputs[i], inputs[i], lr);
            epochLoss += loss;
        }
        std::cout << "epoch " << t << ": loss = " << epochLoss / inputs.size() << ", lr = " << lr << "\n";
        lr *= 0.99f;
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

    printKernel<<<1, 1>>>(inputs.back(), dim);

    std::cout << "\n";

    cudaDeviceSynchronize();

    for (int i = 0; i < inputs.size(); i++) {
        CUDA_CALL(cudaFree(inputs[i]));
    }

    return 0;
}