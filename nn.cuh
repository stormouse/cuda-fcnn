#pragma once

#include <cublas_v2.h>
#include <memory>
#include <vector>

struct FcLayer
{
    FcLayer(int inputDim, int outputDim, cublasHandle_t& handle);
    ~FcLayer();

    float *W;
    float *dW;
    float *b;
    float *db;
    float *z;
    float *a;
    float *input;

    cublasHandle_t& handle;

    int inputDim;
    int outputDim;

    std::vector<float> getWeights();
    std::vector<float> getBias();

    dim3 dimBlock{16};
    dim3 dimGrid{16};

    void feedForward(float *x);
    void backPropagate(float *wNext, float* dNext, int dimNext);
    void applyUpdate(float lr);
};

struct Model
{
    Model(const std::vector<int>& dims);
    ~Model();

    float fit(float *x_device, float *y_device, float lr);

    std::vector<float> getPreviousOutput();

    float *loss_device;
    float previousLoss;

    dim3 dimBlock{16};
    dim3 dimGrid{16};

    cublasHandle_t handle;

    std::vector<float> lossVector;
    std::vector<std::unique_ptr<FcLayer>> layers;
    std::vector<int> dims;
};

__global__ void printKernel(float* data, int len);