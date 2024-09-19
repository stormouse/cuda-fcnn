#include "nn.cuh"
#include "functions.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

__device__ __forceinline__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float d_sigmoid(float x)
{
    float s = sigmoid(x);
    return s * (1 - s);
}

__global__ void sigmoid_kernel(const float* x, float* out, int len)
{
    int step = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += step) {
        out[i] = sigmoid(x[i]);
    }
}

// a .= sigmoid(x)
__global__ void d_sigmoid_mul_kernel(float* a, const float *x, int len)
{
    int step = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += step) {
        a[i] = a[i] * d_sigmoid(x[i]);
    }
}

__global__ void cross_entropy_kernel(const float *x, const float *y, float *out, int len)
{
    int step = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += step) {
        out[i] = -(y[i] * logf(max(x[i], 1e-15f)) + (1 - y[i]) * logf(max(1 - x[i], 1e-15f)));
    }
}

__global__ void d_cross_entropy_kernel(const float *a, const float *y, float *out, int len)
{
    int step = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += step) {
        out[i] = a[i] - y[i];
    }
}

__global__ void printKernel(float* data, int len) {
    int step = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += step) {
        printf("Data[%d] = %f\n", i, data[i]);
    }
}


void FcLayer::feedForward(float *x)
{
    float one = 1.0f;
    float zero = 0.0f;
    this->input = x;

    // z = Wx + b
    cublasSgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outputDim, 1, inputDim,
        &one,
        W, outputDim,
        x, inputDim,
        &zero,
        z, outputDim);
    
    cublasSgeam(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outputDim, 1,
        &one,
        z, outputDim,
        &one,
        b, outputDim,
        z, outputDim);

    // a = sigmoid(z)
    sigmoid_kernel<<<dimGrid,dimBlock>>>(z, a, outputDim);
    CHECK_LAST_CUDA_ERROR();
}

void FcLayer::backPropagate(float *wNext, float* dNext, int dimNext)
{
    float one = 1.0f;
    float zero = 0.0f;

    // db = d = (W_1)^T * d_1 . d_sigmoid(input)
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        outputDim, 1, dimNext,
        &one,
        wNext, dimNext,
        dNext, dimNext,
        &zero,
        db, outputDim));
    
    d_sigmoid_mul_kernel<<<dimGrid,dimBlock>>>(db, input, outputDim);
    CHECK_LAST_CUDA_ERROR();

    // dW = d * input
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        outputDim, inputDim, 1,
        &one,
        db, outputDim,
        input, inputDim,
        &zero,
        dW, outputDim));
}

void FcLayer::applyUpdate(float lr)
{
    float one = 1.0f;
    float step = lr > 0 ? -lr : lr;

    // W = W + dW * lr
    CUBLAS_CHECK(cublasSgeam(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outputDim, inputDim,
        &one,
        W, outputDim,
        &step,
        dW, outputDim,
        W, outputDim));

    // b = b + db * lr
    CUBLAS_CHECK(cublasSgeam(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outputDim, 1,
        &one,
        b, outputDim,
        &step,
        db, outputDim,
        b, outputDim));
}

std::vector<float> FcLayer::getWeights() {
    auto w = std::vector<float>(inputDim * outputDim, 0);
    CUDA_CALL(cudaMemcpy(w.data(), W, inputDim * outputDim * sizeof(float), cudaMemcpyDeviceToHost));
    return w;
}

std::vector<float> FcLayer::getBias() {
    auto bias = std::vector<float>(outputDim, 0);
    CUDA_CALL(cudaMemcpy(bias.data(), b, outputDim * sizeof(float), cudaMemcpyDeviceToHost));
    return bias;
}

FcLayer::FcLayer(int inputDim, int outputDim, cublasHandle_t& handle)
    : inputDim{inputDim}, outputDim{outputDim}, handle{handle}
{
    curandGenerator_t gen;
    int seed = 42;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

    CUDA_CALL(cudaMalloc((void**)&W, inputDim * outputDim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dW, inputDim * outputDim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&b, outputDim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&db, outputDim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&z, outputDim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&a, outputDim * sizeof(float)));
    
    CURAND_CHECK(curandGenerateUniform(gen, W, inputDim * outputDim));
    CURAND_CHECK(curandGenerateUniform(gen, b, outputDim));

    float rr = 0.1f;
    CUBLAS_CHECK(cublasSscal(this->handle, inputDim * outputDim, &rr, W, 1));
    CUBLAS_CHECK(cublasSscal(this->handle, outputDim, &rr, b, 1));

    cudaDeviceSynchronize();

    printKernel<<<1, 1>>>(W, inputDim * outputDim);
}

FcLayer::~FcLayer()
{
    CUDA_CALL(cudaFree(W));
    CUDA_CALL(cudaFree(dW));
    CUDA_CALL(cudaFree(b));
    CUDA_CALL(cudaFree(db));
    CUDA_CALL(cudaFree(z));
    CUDA_CALL(cudaFree(a));
}

Model::Model(const std::vector<int>& dims)
    : dims{dims}
{
    cublasCreate(&handle);

    for (int i = 1; i < dims.size(); i++)
    {
        auto l = std::make_unique<FcLayer>(dims[i - 1], dims[i], handle);
        layers.emplace_back(std::move(l));
    }

    int inDim = dims[0];
    int outDim = dims[dims.size() - 1];

    CUDA_CALL(cudaMalloc((void**)&loss_device, outDim * sizeof(float)));
}

Model::~Model()
{
    CUDA_CALL(cudaFree(loss_device));
    cublasDestroy(handle);
}

float Model::fit(float *x_device, float *y_device, float lr)
{
    float one = 1.0f;
    float zero = 0.0f;

    layers[0]->feedForward(x_device);
    for (int i = 1; i < layers.size(); i++) {
        layers[i]->feedForward(layers[i - 1]->a);
    }

    int lastDim = layers.back()->outputDim;
    lossVector.resize(lastDim);

    cross_entropy_kernel<<<dimGrid, dimBlock>>>(layers.back()->a, y_device, loss_device, lastDim);
    CHECK_LAST_CUDA_ERROR();
    d_cross_entropy_kernel<<<dimGrid, dimBlock>>>(layers.back()->a, y_device, layers.back()->db, lastDim);
    CHECK_LAST_CUDA_ERROR();
    d_sigmoid_mul_kernel<<<dimGrid,dimBlock>>>(layers.back()->db, layers.back()->a, lastDim);
    CHECK_LAST_CUDA_ERROR();

    // dW = d * input
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        lastDim, dims[dims.size() - 1], 1,
        &one,
        layers.back()->db, lastDim,
        layers.size() > 2 ? layers[layers.size() - 2]->a : x_device, dims[dims.size() - 1],
        &zero,
        layers.back()->dW, lastDim));


    for (int i = layers.size() - 2; i >= 0; i--)
    {
        layers[i]->backPropagate(layers[i + 1]->W, layers[i + 1]->db, dims[i + 1]);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->applyUpdate(lr);
    }

    // reduce<256><<<dimGrid, dimBlock>>>(error_device, loss_device);

    cudaDeviceSynchronize();

    // CUDA_CALL(cudaMemcpy(&previousLoss, loss_device, sizeof(float), cudaMemcpyDeviceToHost));
    cudaMemcpy(lossVector.data(), loss_device, lastDim * sizeof(float), cudaMemcpyDeviceToHost);

    auto loss = 0.0f;
    for (auto v : lossVector) {
        loss += v;
    }

    loss /= lossVector.size();
    previousLoss = loss;

    return loss;
}

std::vector<float> Model::getPreviousOutput()
{
    auto v = std::vector<float>(dims.back(), 0);
    printKernel<<<1, 1>>>(layers.back()->a, 64);
    cudaDeviceSynchronize();
    CUDA_CALL(cudaMemcpy(v.data(), layers.back()->a, dims.back() * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}

// // optimized reduce
// // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// template <unsigned int blockSize>
// __device__ void warpReduce(volatile float *sdata, unsigned int tid) {
//     if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
//     if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
//     if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
//     if (blockSize >=   8) sdata[tid] += sdata[tid +  4];
//     if (blockSize >=   4) sdata[tid] += sdata[tid +  2];
//     if (blockSize >=   2) sdata[tid] += sdata[tid +  1];
// }

// template <unsigned int blockSize>
// __global__ void reduce(float *g_idata, float *g_odata, unsigned int n) {
//     extern __shared__ float sdata[];
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x * (blockSize * 2) + tid;
//     unsigned int gridSize = blockSize * 2 * gridDim.x;
//     sdata[tid] = 0;
//     while (i < n) { sdata[tid] += g_idata[i] + g_idata[i + blockSize];  i += gridSize;  }
//     __syncthreads();
//     if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
//     if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
//     if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
//     if (tid < 32) warpReduce(sdata, tid);
//     if (tid == 0) g_odata[blockIdx.x] = sdata[0];
//  }