#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cufft.h"

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

// conv_im2col
template <typename Dtype>
__global__ void im2colKern(const int n, const Dtype *pic, const int height, const int width, const int ksize, 
                           const int pad, const int stride, const int colHeight, const int colWidth, Dtype* colPic) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for( ; i < n; i += blockDim.x * gridDim.x)
    {
        int outWidth = i % colWidth;
        i = i / colWidth;
        int outHeight = i % colHeight;
        int inChan = i / colHeight;
        int outChan = inChan * ksize * ksize;
        int inHeight = outHeight * stride - pad;
        int inWidth = outWidth * stride - pad;
        colPic += (outChan * colHeight + outHeight) * colWidth + outWidth;
        pic += (inChan * height + inHeight) * width + inWidth;
        for (int p = 0; p < ksize; p++) 
        {
            for (int q = 0; q < ksize; q++) 
            {
                *colPic = (inHeight + p >= 0 && inWidth + q >= 0 && inHeight + p < height && inWidth + q < width) ? 
                          pic[p * width + q] : 0;
                colPic += colHeight * colWidth;
            }
        }
    }
}

template <typename Dtype>
__global__ void col2imKern(const int n, const Dtype* colPic, const int height, const int width, 
                           const int channels, const int ksize, const int pad, const int stride, 
                           const int colHeight, const int colWidth, Dtype* pic) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for( ; i < n; i++)
    {
        Dtype val = 0;
        int w = i % width + pad;
        int h = (i / width) % height + pad;
        int c = i / (width * height);
        int colStartW = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        int colEndW = min(w / stride + 1, colWidth);
        int colStartH = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        int colEndH = min(h / stride + 1, colHeight);
        for (int p = colStartH; p < colEndH; p++) 
        {
            for (int q = colStartW; q < colEndW; q++) 
            {
                int tmp = c * ksize * ksize + (h - p * stride) * ksize + (w - q * stride);
                val = colPic[(tmp * colHeight + p) * colWidth + q];
            }
        }
        pic[i] = val;
  }
}

void im2colConvHost(float *pic, const int height, const int width, float *kern, const int ksize, 
                    const int channels, const int pad, const int stride)
{
    float *devP, *devC, *devK, *devKC;

    int size = height * width * sizeof(float);
    cudaMalloc((void **)&devP, size);
    cudaMemcpy(devP, pic, size, cudaMemcpyHostToDevice);
    int kksize = ksize * ksize * sizeof(float);
    cudaMalloc((void **)&devK, kksize);
    cudaMemcpy(devK, kern, kksize, cudaMemcpyHostToDevice);

    // image to colImage
    int colHeight = (height + 2 * pad - ksize) / stride + 1;
    int colWidth = (width + 2 * pad - ksize) / stride + 1;
    int csize = colHeight * colWidth * channels * ksize * ksize * sizeof(float);
    cudaMalloc((void **)&devC, csize);
    int nkern = channels * colHeight * colWidth;
    im2colKern<float><<<GET_BLOCK_NUM(nkern), DEFAULT_THREAD_NUM>>>(nkern, devP, height, width, ksize, pad, stride, 
                                                                    colHeight, colWidth, devC);
    // test
    float *colPic = new float[csize];
    cudaMemcpy(colPic, devC, csize, cudaMemcpyDeviceToHost);
    for(int i = 0; i < channels * ksize * ksize; i++)
    {
        for(int j = 0; j < colWidth * colHeight; j++)
        {
            printf("%.2lf ", colPic[i * colWidth * colHeight + j]);
        }
        printf("\n");
    }

    // kernel to colKernel
    int colKernSize = (ksize + 2 * pad - ksize) / stride + 1;
    int cksize = colKernSize * colKernSize * channels * ksize * ksize * sizeof(float);
    cudaMalloc((void **)&devKC, cksize);
    nkern = channels * colKernSize * colKernSize;
    im2colKern<float><<<GET_BLOCK_NUM(nkern), DEFAULT_THREAD_NUM>>>(nkern, devK, ksize, ksize, ksize, 0, 1, 
                                                                    colKernSize, colKernSize, devKC);
    // test
    float * colKern = new float(cksize);
    cudaMemcpy(colKern, devKC, cksize, cudaMemcpyDeviceToHost);
    printf("\n");
    for(int i = 0; i < channels * ksize * ksize; i++)
    {
        for(int j = 0; j < colKernSize * colKernSize; j++)
        {
            printf("%.2lf ", colKern[i * colKernSize * colKernSize + j]);
        }
        printf("\n");
    }

    // GEMM
    float alpha = 1.0, beta = 0.0;
    cublasHandle_t handle;
    int status = cublasCreate(&handle);
    int k = channels * ksize * ksize;
    int n = colWidth * colWidth;
    int m = colKernSize * colKernSize;
    float *mres, *res;
    cudaMalloc((void **)&mres, n * k * sizeof(float));
    cudaMalloc((void **)&res, n * k * sizeof(float));
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, 
                         &alpha, devC, n, devKC, m, &beta, mres, n);
    res = new float[n * k * sizeof(float)];
    cudaMemcpy(res, mres, n * k * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\n");
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            printf("%.2lf ", pr[i * n + j]);
        }
        printf("\n");
    }
    status = cublasDestroy(handle);

    // col2im

    cudaFree(devP);
    cudaFree(devC);
    cudaFree(devK);
    cudaFree(devKC);
    cudaFree(result);
    cudaFree(imRes);
}

void im2colHost(float *pic, const int height, const int width, const int channels, 
                const int ksize, const int pad, const int stride)
{
    float *devP, *devC;

    int size = height * width * sizeof(float);
    cudaMalloc((void **)&devP, size);
    cudaMemcpy(devP, pic, size, cudaMemcpyHostToDevice);

    int colHeight = (height + 2 * pad - ksize) / stride + 1;
    int colWidth = (width + 2 * pad - ksize) / stride + 1;
    int csize = colHeight * colWidth * channels * ksize * ksize * sizeof(float);
    cudaMalloc((void **)&devC, csize);

    int nkern = channels * colHeight * colWidth;
    im2colKern<float><<<GET_BLOCK_NUM(nkern), DEFAULT_THREAD_NUM>>>(nkern, devP, height, width, ksize, pad, stride, 
                                                                    colHeight, colWidth, devC);
    float *colPic = new float[csize];
    cudaMemcpy(colPic, devC, csize, cudaMemcpyDeviceToHost);
    for(int i = 0; i < colHeight * colWidth; i++)
    {
        for(int j = 0; j < channels * ksize * ksize; j++)
        {
            printf("%.2lf ", colPic[i * colHeight * colWidth + j]);
        }
        printf("\n");
    }
    float *imPic = new float[size];
    nkern = channels * height * width;
    col2imKern<float><<<GET_BLOCK_NUM(nkern), DEFAULT_THREAD_NUM>>>(nkern, devC, height, width, channels, ksize, pad, stride, 
                                                                    colHeight, colWidth, devP);
    cudaMemcpy(imPic, devP, size, cudaMemcpyDeviceToHost);
    printf("\n");
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            printf("%.2lf ", imPic[i * width + j]);
        }
        printf("\n");
    }
    cudaFree(devP);
    cudaFree(devC);
}

void im2colHost(double *pic, const int height, const int width, const int channels, 
                const int ksize, const int pad, const int stride)
{
    double *devP, *devC;
    
    int size = height * width * sizeof(double);
    cudaMalloc((void **)&devP, size);
    cudaMemcpy(devP, pic, size, cudaMemcpyHostToDevice);
    
    int colHeight = (height + 2 * pad - ksize) / stride + 1;
    int colWidth = (width + 2 * pad - ksize) / stride + 1;
    int csize = colHeight * colWidth * channels * ksize * ksize * sizeof(double);
    cudaMalloc((void **)&devC, csize);
    
    int nkern = channels * colHeight * colWidth;
    im2colKern<double><<<GET_BLOCK_NUM(nkern), DEFAULT_THREAD_NUM>>>(nkern, devP, height, width, ksize, pad, stride, 
                                                                    colHeight, colWidth, devC);
    double *colPic = new double[csize];
    cudaMemcpy(colPic, devC, csize, cudaMemcpyDeviceToHost);
    for(int i = 0; i < colHeight * colWidth; i++)
    {
        for(int j = 0; j < channels * ksize * ksize; j++)
        {
            printf("%.2lf ", colPic[i * colHeight * colWidth + j]);
        }
        printf("\n");
    }
    
    cudaFree(devP);
    cudaFree(devC);
}


__global__ void paddingKern()
{

}

void padding()
{

}

__global__ void complexVectorDotKern(cufftComplex *A, cufftComplex *B, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for( ; i < n; i++)
    {
        A[i].x = A[i].x * B[i].x - A[i].y * B[i].y;
        A[i].y = A[i].x * B[i].y + A[i].y * B[i].x;
    }
}

void matrixConvFFT(float *padPic, float *padKern, int height, int width)
{        
    cufftReal *realDevP, *realDevK;
    cufftComplex *cplDevP, *cplDevK;
    int rsize = width * height * sizeof(cufftReal);
    int csize = width * (height / 2 + 1) * sizeof(cufftComplex);
    
    cudaMalloc((void**)&realDevP, rsize);
    cudaMalloc((void**)&realDevK, rsize);

    cudaMalloc((void**)&cplDevP, csize);
    cudaMalloc((void**)&cplDevK, csize);

    cudaMemset(realDevP, 0, rsize);
    cudaMemset(realDevK, 0, rsize);

    cudaMemcpy(realDevP, padPic, rsize,	cudaMemcpyHostToDevice);
    cudaMemcpy(realDevK, padKern, rsize, cudaMemcpyHostToDevice);
    
    // picture, kernel, result
    cufftHandle planP, planK, planR;
    cufftPlan2d(&planP, height, width, CUFFT_R2C);
    cufftPlan2d(&planK, height, width, CUFFT_R2C);
    cufftPlan2d(&planR, height, width, CUFFT_C2R);
    
    cufftExecR2C(planP, realDevP, cplDevP);
    cufftExecR2C(planK, realDevK, cplDevK);
    
    complexVectorDotKern<<<GET_BLOCK_NUM(ceil(width * (height / 2 + 1))), DEFAULT_THREAD_NUM>>>(cplDevP, cplDevK, width * (height / 2 + 1));

    cufftExecC2R(planR, cplDevP, realDevP);
    
    cufftReal* result = new cufftReal[width * (height / 2 + 1) * 2];
    cudaMemcpy(result, realDevP, rsize, cudaMemcpyDeviceToHost);
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            printf("%.2lf ", (1.0f / (width * height)) * result[i * width + j]);
        }
        printf("\n");
    }

    cudaFree(realDevP);
    cudaFree(realDevK);
    cudaFree(cplDevP);
    cudaFree(cplDevK);

    cufftDestroy(planP);
    cufftDestroy(planK);
    cufftDestroy(planR);
}

void matrixConvIm2col()
{
    // im2col: picture, kernel
    // gemm
    // col2im: result
}

void matrixConvFFTHost()
{
    // padding
    // conv
    // unpadding
}

// conv_winograd

int main()
{
    // float a[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    // float b[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    // matrixConvFFT(a, b, 3, 3);
    float c[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float k[] = {1, 3, 5, 7};
    im2colConvHost(c, 3, 3, k, 2, 1, 0, 1);
}



