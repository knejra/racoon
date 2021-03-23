#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#define DEFAULT_THREAD_NUM 64
#define GET_BLOCK_NUM(n) (((n) + DEFAULT_THREAD_NUM - 1) / DEFAULT_THREAD_NUM)

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

template<typename Dtype>
__global__ void scalarAddKern(Dtype *dst, const Dtype scalar, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(; i < n; i++)
    {
        dst[i] += scalar;
    }
}

void scalarAdd(float *dst, const float scalar, int n)
{
    scalarAddKern<float><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(dst, scalar, n);
}

void scalarAdd(double *dst, const double scalar, int n)
{
    scalarAddKern<double><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(dst, scalar, n);
}

void scalarAddHost(float *dst, const float scalar, int n)
{
    float *devs;
    cudaMalloc((void **)&devs, n * sizeof(float));
    cudaMemcpy(devs, dst, n * sizeof(float), cudaMemcpyHostToDevice);
    scalarAdd(devs, scalar, n);
    cudaMemcpy(dst, devs, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devs);
}

void scalarAddHost(double *dst, const double scalar, int n)
{
    double *devs;
    cudaMalloc((void **)&devs, n * sizeof(double));
    cudaMemcpy(devs, dst, n * sizeof(double), cudaMemcpyHostToDevice);
    scalarAdd(devs, scalar, n);
    cudaMemcpy(dst, devs, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(devs);
}

template<typename Dtype>
__global__ void scalarSubKern(Dtype *dst, const Dtype scalar, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(; i < n; i++)
    {
        dst[i] -= scalar;
    }
}

void scalarSub(float *dst, const float scalar, int n)
{
    scalarSubKern<float><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(dst, scalar, n);
}

void scalarSub(double *dst, const double scalar, int n)
{
    scalarSubKern<double><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(dst, scalar, n);
}

void scalarSubHost(float *dst, const float scalar, int n)
{
    float *devs;
    cudaMalloc((void **)&devs, n * sizeof(float));
    cudaMemcpy(devs, dst, n * sizeof(float), cudaMemcpyHostToDevice);
    scalarSub(devs, scalar, n);
    cudaMemcpy(dst, devs, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devs);
}

void scalarSubHost(double *dst, const double scalar, int n)
{
    double *devs;
    cudaMalloc((void **)&devs, n * sizeof(double));
    cudaMemcpy(devs, dst, n * sizeof(double), cudaMemcpyHostToDevice);
    scalarSub(devs, scalar, n);
    cudaMemcpy(dst, devs, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(devs);
}

template<typename Dtype>
__global__ void matrixAddKern(Dtype *A, Dtype *B, Dtype *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for( ; i < n; i++)
    {
        C[i] = A[i] + B[i];
    }
}

void matrixAdd(float *A, float *B, float *C, int n)
{
    matrixAddKern<float><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(A, B, C, n);
}

void matrixAdd(double *A, double *B, double *C, int n)
{
    matrixAddKern<double><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(A, B, C, n);
}

void matrixAddHost(float *A, float *B, float *C, int n)
{
    float *devA, *devB, *devC;
    cudaMalloc((void **)&devA, n * sizeof(float));
    cudaMalloc((void **)&devB, n * sizeof(float));
    cudaMalloc((void **)&devC, n * sizeof(float));
    cudaMemcpy(devA, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, n * sizeof(float), cudaMemcpyHostToDevice);
    matrixAdd(devA, devB, devC, n);
    cudaMemcpy(C, devC, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}

void matrixAddHost(double *A, double *B, double *C, int n)
{
    double *devA, *devB, *devC;
    cudaMalloc((void **)&devA, n * sizeof(double));
    cudaMalloc((void **)&devB, n * sizeof(double));
    cudaMalloc((void **)&devC, n * sizeof(double));
    cudaMemcpy(devA, A, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, n * sizeof(double), cudaMemcpyHostToDevice);
    matrixAdd(devA, devB, devC, n);
    cudaMemcpy(C, devC, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}

template<typename Dtype>
__global__ void matrixSubKern(Dtype *A, Dtype *B, Dtype *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for( ; i < n; i++)
    {
        C[i] = A[i] - B[i];
    }
}

void matrixSub(float *A, float *B, float *C, int n)
{
    matrixSubKern<float><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(A, B, C, n);
}

void matrixSub(double *A, double *B, double *C, int n)
{
    matrixSubKern<double><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(A, B, C, n);
}

void matrixSubHost(float *A, float *B, float *C, int n)
{
    float *devA, *devB, *devC;
    cudaMalloc((void **)&devA, n * sizeof(float));
    cudaMalloc((void **)&devB, n * sizeof(float));
    cudaMalloc((void **)&devC, n * sizeof(float));
    cudaMemcpy(devA, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, n * sizeof(float), cudaMemcpyHostToDevice);
    matrixSub(devA, devB, devC, n);
    cudaMemcpy(C, devC, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}

void matrixSubHost(double *A, double *B, double *C, int n)
{
    double *devA, *devB, *devC;
    cudaMalloc((void **)&devA, n * sizeof(double));
    cudaMalloc((void **)&devB, n * sizeof(double));
    cudaMalloc((void **)&devC, n * sizeof(double));
    cudaMemcpy(devA, A, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, n * sizeof(double), cudaMemcpyHostToDevice);
    matrixSub(devA, devB, devC, n);
    cudaMemcpy(C, devC, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}

template<typename Dtype>
__global__ void vectorDotKern(Dtype *A, Dtype *B, Dtype *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for( ; i < n; i++)
    {
        C[i] = A[i] * B[i];
    }
}

void vectorDot(float *A, float *B, float *C, int n)
{
    vectorDotKern<float><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(A, B, C, n);
}

void vectorDot(double *A, double *B, double *C, int n)
{
    vectorDotKern<double><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(A, B, C, n);
}

void vectorDotHost(float *A, float *B, float *C, int n)
{
    float *devA, *devB, *devC;
    cudaMalloc((void **)&devA, n * sizeof(float));
    cudaMalloc((void **)&devB, n * sizeof(float));
    cudaMalloc((void **)&devC, n * sizeof(float));
    cudaMemcpy(devA, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, n * sizeof(float), cudaMemcpyHostToDevice);
    vectorDot(devA, devB, devC, n);
    cudaMemcpy(C, devC, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}

void vectorDotHost(double *A, double *B, double *C, int n)
{
    double *devA, *devB, *devC;
    cudaMalloc((void **)&devA, n * sizeof(double));
    cudaMalloc((void **)&devB, n * sizeof(double));
    cudaMalloc((void **)&devC, n * sizeof(double));
    cudaMemcpy(devA, A, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, n * sizeof(double), cudaMemcpyHostToDevice);
    vectorDot(devA, devB, devC, n);
    cudaMemcpy(C, devC, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}

template<typename Dtype>
__global__ void vectorDivKern(Dtype *A, Dtype *B, Dtype *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for( ; i < n; i++)
    {
        C[i] = A[i] / B[i];
    }
}

void vectorDiv(float *A, float *B, float *C, int n)
{
    vectorDivKern<float><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(A, B, C, n);
}

void vectorDiv(double *A, double *B, double *C, int n)
{
    vectorDivKern<double><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(A, B, C, n);
}

void vectorDivHost(float *A, float *B, float *C, int n)
{
    float *devA, *devB, *devC;
    cudaMalloc((void **)&devA, n * sizeof(float));
    cudaMalloc((void **)&devB, n * sizeof(float));
    cudaMalloc((void **)&devC, n * sizeof(float));
    cudaMemcpy(devA, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, n * sizeof(float), cudaMemcpyHostToDevice);
    vectorDiv(devA, devB, devC, n);
    cudaMemcpy(C, devC, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}

void vectorDivHost(double *A, double *B, double *C, int n)
{
    double *devA, *devB, *devC;
    cudaMalloc((void **)&devA, n * sizeof(double));
    cudaMalloc((void **)&devB, n * sizeof(double));
    cudaMalloc((void **)&devC, n * sizeof(double));
    cudaMemcpy(devA, A, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, n * sizeof(double), cudaMemcpyHostToDevice);
    vectorDiv(devA, devB, devC, n);
    cudaMemcpy(C, devC, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}

void matrixGemmHost(float *A, float *B, float *C, int m, int n, int k)
{
    float *devA, *devB, *devC;
    float alpha = 1.0, beta = 0.0;
    cublasHandle_t handle;
    int status = cublasCreate(&handle);
    int asize = m * k;
    int bsize = k * n;
    int csize = m * n;
    cudaMalloc((void **)&devA, asize * sizeof(float));
    cudaMalloc((void **)&devB, bsize * sizeof(float));
    cudaMalloc((void **)&devC, csize * sizeof(float));
    status = cublasSetVector(asize, sizeof(float), A, 1, devA, 1);
    status = cublasSetVector(bsize, sizeof(float), B, 1, devB, 1);
    status = cublasSetVector(csize, sizeof(float), C, 1, devC, 1);
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, 
                         &alpha, devB, n, devA, k, &beta, devC, n);
    status = cublasGetVector(csize, sizeof(float), devC, 1, C, 1);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    status = cublasDestroy(handle);
}

void matrixGemmHost(double *A, double *B, double *C, int m, int n, int k)
{
    double *devA, *devB, *devC;
    double alpha = 1.0, beta = 0.0;
    cublasHandle_t handle;
    int status = cublasCreate(&handle);
    int asize = m * k;
    int bsize = k * n;
    int csize = m * n;
    cudaMalloc((void **)&devA, asize * sizeof(double));
    cudaMalloc((void **)&devB, bsize * sizeof(double));
    cudaMalloc((void **)&devC, csize * sizeof(double));
    status = cublasSetVector(asize, sizeof(double), A, 1, devA, 1);
    status = cublasSetVector(bsize, sizeof(double), B, 1, devB, 1);
    status = cublasSetVector(csize, sizeof(double), C, 1, devC, 1);
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, 
                         &alpha, devB, n, devA, k, &beta, devC, n);
    status = cublasGetVector(csize, sizeof(double), devC, 1, C, 1);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    status = cublasDestroy(handle);
}

// float, C = A_T * B
void matrixTransGemmHost(float *A, float *B, float *C, int m, int n, int k)
{
    float *devA, *devB, *devC;
    float alpha = 1.0, beta = 0.0;
    cublasHandle_t handle;
    int status = cublasCreate(&handle);
    int asize = m * k;
    int bsize = k * n;
    int csize = m * n;
    cudaMalloc((void **)&devA, asize * sizeof(float));
    cudaMalloc((void **)&devB, bsize * sizeof(float));
    cudaMalloc((void **)&devC, csize * sizeof(float));
    status = cublasSetVector(asize, sizeof(float), A, 1, devA, 1);
    status = cublasSetVector(bsize, sizeof(float), B, 1, devB, 1);
    status = cublasSetVector(csize, sizeof(float), C, 1, devC, 1);
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, 
                         &alpha, devB, n, devA, m, &beta, devC, n);
    status = cublasGetVector(csize, sizeof(float), devC, 1, C, 1);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    status = cublasDestroy(handle);
}

// double, C = A_T * B
void matrixTransGemmHost(double *A, double *B, double *C, int m, int n, int k)
{
    double *devA, *devB, *devC;
    double alpha = 1.0, beta = 0.0;
    cublasHandle_t handle;
    int status = cublasCreate(&handle);
    int asize = m * k;
    int bsize = k * n;
    int csize = m * n;
    cudaMalloc((void **)&devA, asize * sizeof(double));
    cudaMalloc((void **)&devB, bsize * sizeof(double));
    cudaMalloc((void **)&devC, csize * sizeof(double));
    status = cublasSetVector(asize, sizeof(double), A, 1, devA, 1);
    status = cublasSetVector(bsize, sizeof(double), B, 1, devB, 1);
    status = cublasSetVector(csize, sizeof(double), C, 1, devC, 1);
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, 
                         &alpha, devB, n, devA, m, &beta, devC, n);
    status = cublasGetVector(csize, sizeof(double), devC, 1, C, 1);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    status = cublasDestroy(handle);
}

int main()
{
    double a[] = {1.0, 2.0, 3.0, 4.0};
    double b[] = {2.0, 3.0, 4.0, 5.0};
    double c[4] = {0};
    matrixTransGemmHost(a, b, c, 2, 2, 2);
    for(int i = 0; i < 4; i++)
    {
        printf("%.2lf\n", c[i]);
    } 
}

// conv_im2col
// conv_fft



