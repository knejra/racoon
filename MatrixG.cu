#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "cublas_v2.h"

#include "Matrix.h"

template<typename Dtype>
__global__ void scalarAddKern(Dtype *dst, const Dtype scalar, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        dst[i] += scalar;
    }
}

template<typename Dtype>
void scalarAdd(float *dst, const float scalar, int n)
{
    scalarAddKern<float><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(dst, scalar, n);
}

template<typename Dtype>
void scalarAdd(Matrix<Dtype> &m, const Dtype scalar)
{
    Dtype *darray;
    cudaMalloc((void **)&darray, m.dimension() * sizeof(Dtype));
    int ret = cublasSetVector(m.dimension(), sizeof(Dtype), m.array, 1, darray, 1);
    scalarAdd(darray, scalar, m.dimension());
    ret = cublasGetVector(m.dimension(), sizeof(Dtype), darray, 1, m.array, 1);
    cudaFree(darray);
}

for test
template<typename Dtype>
void naiveScalarAdd(Matrix<Dtype> &m, const Dtype scalar)
{
    for(int i = 0; i < m.dimension(); i++)
    {
        m.array[i] += scalar;
    }
}

template<typename Dtype>
void compare(Matrix<Dtype> ma, Matrix<Dtype> mb)
{
    float normErr, diff;
    for(int i = 0; i < ma.dimension(); i++)
    {
        diff += abs(ma.array[i] - mb.array[i]);
        normErr += diff * diff;
    }
    printf("difference: %lf, norm error: %lf\n", diff, normErr);
}

void scalarAddTest()
{
    Matrix<float> rm(100, 80, 1.0, 9.0, M_RAND);
    Matrix<float> rm1 = rm;
    float adder = 3.6;
    naiveScalarAdd(rm, adder);
    scalarAdd(rm1, adder);
    compare(rm, rm1);
}

template<typename Dtype>
__global__ void scalarSubKern(Dtype *dst, const Dtype scalar, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        dst[i] -= scalar;
    }
}

void scalarSub(float *dst, const float scalar, int n)
{
    scalarAddKern<float><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(dst, scalar, n);
}

void scalarSub(double *dst, const double, scalar, int n)
{
    scalarSubKern<double><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(dst, scalar, n);
}

// scalarSub: host interface
void scalarSub(Matrix<float> *m, const float scalar)
{
    float *darray;
    int ret = -1;
    cudaMalloc((void **)&darray, m->dimension() * sizeof(float));
    ret = cublasSetVector(m->dimension(), sizeof(float), m->array, 1, darray, 1);
    scalarSub(darray, scalar, m->dimension());
    ret = cublasGetVector(m->dimension(), sizeof(float), darray, 1, m->array, 1);
    cudaFree(darray);
}

void scalarSub(Matrix<double> *m, const double scalar)
{
    double *darray;
    int ret = -1;
    cudaMalloc((void **)&darray, m->dimension() * sizeof(double));
    ret = cublasSetVector(m->dimension(), sizeof(double), m->array, 1, darray, 1);
    scalarSub(darray, scalar, m->dimension());
    ret = cublasGetVector(m->dimension(), sizeof(double), darray, 1, m->array, 1);
    cudaFree(darray);
}

// for test
void naiveScalarSub(Matrix<float> *m, const float scalar)
{
    for(int i = 0; i < m->dimension(); i++)
    {
        m->array[i] += scalar;
    }
}

void scalarAddTest()
{
    Matrix<float> rm(100, 80, 1.0, 9.0);
    Matrix<float> rm1 = rm;
    naiveScalarSub(&rm, 3.6)l
    scalarSub(&rm1, 3.6);
    compare(&rm1, &rm2);
}

template<typename Dtype>
__global__ void matrixAddKern(Dtype *A, Dtype *B, Dtype *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
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

void matrixAdd(Matrix<float> *ma, Matrix<float> *mb, Matrix<float> *mc)
{
    CHECK(ma->dimension() == mb->dimension());
    CHECK(mb->dimension() == mc->dimension());
    float *da, *db, *dc;
    int ret = -1;
    cudaMalloc((void **)&da, ma->dimension() * sizeof(float));
    cudaMalloc((void **)&db, mb->dimension() * sizeof(float));
    cudaMalloc((void **)&dc, mc->dimension() * sizeof(float));
    ret = cublasSetVector(ma->dimension(), sizeof(float), ma->array, 1, da, 1);
    ret = cublasSetVector(mb->dimension(), sizeof(float), mb->array, 1, db, 1);
    matrixAdd(da, db, dc, mc->dimension());
    ret = cublasGetVector(mc->dimension(), sizeof(float), dc, 1, mc->array, 1);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

void matrixAdd(Matrix<double> *ma, Matrix<double> *mb, Matrix<double> *mc)
{
    CHECK(ma->dimension() == mb->dimension());
    CHECK(mb->dimension() == mc->dimension());
    double *da, *db, *dc;
    int ret = -1;
    cudaMalloc((void **)&da, ma->dimension() * sizeof(double));
    cudaMalloc((void **)&db, mb->dimension() * sizeof(double));
    cudaMalloc((void **)&dc, mc->dimension() * sizeof(double));
    ret = cublasSetVector(ma->dimension(), sizeof(double), ma->array, 1, da, 1);
    ret = cublasSetVector(mb->dimension(), sizeof(double), mb->array, 1, db, 1);
    matrixAdd(da, db, dc, mc->dimension());
    ret = cublasGetVector(mc->dimension(), sizeof(double), dc, 1, mc->array, 1);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

void matrixAddTest()
{
    Matrix<float> a(100, 80, 1.0, 9.0);
    Matrix<float> b(100, 80, 1.0, 9.0);
    Matrix<float> rc;
    rc = a + b;
    Matrix<float> c;
    matrixAdd(&a, &b, &c);
    compare(&c, &rc);
}

template<typename Dtype>
__global__ void matrixSubKern(Dtype *A, Dtype *B, Dtype *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
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
    maxtrixSubKern<double><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(A, B, C, n);
}

void matrixSub(Matrix<float> *ma, Matrix<float> *mb, Matrix<float> *mc)
{
    CHECK(ma->dimension() == mb->dimension());
    CHECK(mb->dimension() == mc->dimension());
    float *da, *db, *dc;
    int ret = -1;
    cudaMalloc((void **)&da, ma->dimension() * sizeof(float));
    cudaMalloc((void **)&db, mb->dimension() * sizeof(float));
    cudaMalloc((void **)&dc, mc->dimension() * sizeof(float));
    ret = cublasSetVector(ma->dimension(), sizeof(float), ma->array, 1, da, 1);
    ret = cublasSetVector(mb->dimension(), sizeof(float), mb->array, 1, db, 1);
    matrixSub(da, db, dc, mc->dimension());
    ret = cublasGetVector(mc->dimension(), sizeof(float), dc, 1, mc->array, 1);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

void matrixSub(Matrix<double> *ma, Matrix<double> *mb, Matrix<double> *mc)
{
    CHECK(ma->dimension() == mb->dimension());
    CHECK(mb->dimension() == mc->dimension());
    double *da, *db, *dc;
    int ret = -1;
    cudaMalloc((void **)&da, ma->dimension() * sizeof(double));
    cudaMalloc((void **)&db, mb->dimension() * sizeof(double));
    cudaMalloc((void **)&dc, mc->dimension() * sizeof(double));
    ret = cublasSetVector(ma->dimension(), sizeof(double), ma->array, 1, da, 1);
    ret = cublasSetVector(mb->dimension(), sizeof(double), mb->array, 1, db, 1);
    matrixSub(da, db, dc, mc->dimension());
    ret = cublasGetVector(mc->dimension(), sizeof(double), dc, 1, mc->array, 1);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

void matrixSubTest()
{
    Matrix<float> a(100, 80, 1.0, 9.0);
    Matrix<float> b(100, 80, 1.0, 9.0);
    Matrix<float> rc;
    rc = a + b;
    Matrix<float> c;
    matrixSub(&a, &b, &c);
    compare(&c, &rc);
}

template<typename Dtype>
__global__ void vectorDotKern(Dtype *A, Dtype *B, Dtype *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
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

void vectorDot(Matrix<float> *ma, Matrix<float> *mb, Matrix<float> *mc)
{
    CHECK(ma->dimension() == mb->dimension());
    CHECK(mb->dimension() == mc->dimension());
    float *da, *db, *dc;
    int ret = -1;
    cudaMalloc((void **)&da, ma->dimension() * sizeof(float));
    cudaMalloc((void **)&db, mb->dimension() * sizeof(float));
    cudaMalloc((void **)&dc, mc->dimension() * sizeof(float));
    ret = cublasSetVector(ma->dimension(), sizeof(float), ma->array, 1, da, 1);
    ret = cublasSetVector(mb->dimension(), sizeof(float), mb->array, 1, db, 1);
    vectorDot(da, db, dc, mc->dimension());
    ret = cublasGetVector(mc->dimension(), sizeof(float), dc, 1, mc->array, 1);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

void vectorDot(Matrix<double> *ma, Matrix<double> *mb, Matrix<double> *mc)
{
    CHECK(ma->dimension() == mb->dimension());
    CHECK(mb->dimension() == mc->dimension());
    double *da, *db, *dc;
    int ret = -1;
    cudaMalloc((void **)&da, ma->dimension() * sizeof(double));
    cudaMalloc((void **)&db, mb->dimension() * sizeof(double));
    cudaMalloc((void **)&dc, mc->dimension() * sizeof(double));
    ret = cublasSetVector(ma->dimension(), sizeof(double), ma->array, 1, da, 1);
    ret = cublasSetVector(mb->dimension(), sizeof(double), mb->array, 1, db, 1);
    matrixDot(da, db, dc, mc->dimension());
    ret = cublasGetVector(mc->dimension(), sizeof(double), dc, 1, mc->array, 1);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

template<typename Dtype>
__global__ void matrixDivKern(Dtype *A, Dtype *B, Dtype *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n && B[i] != 0)
    {
        C[i] = A[i] / B[i];
    }
}

void matrixDiv(float *A, float *B, float *C, int n)
{
    matrixDivKern<float><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(A, B, C, n);
}

void matrixDiv(double *A, double *B, double *C, int n)
{
    matrixDivKern<double><<<GET_BLOCK_NUM(n), DEFAULT_THREAD_NUM>>>(A, B, C, n);
}

void matrixDiv(Matrix<float> *ma, Matrix<float> *mb, Matrix<float> *mc)
{
    CHECK(ma->dimension() == mb->dimension());
    CHECK(mb->dimension() == mc->dimension());
    float *da, *db, *dc;
    int ret = -1;
    cudaMalloc((void **)&da, ma->dimension() * sizeof(float));
    cudaMalloc((void **)&db, mb->dimension() * sizeof(float));
    cudaMalloc((void **)&dc, mc->dimension() * sizeof(float));
    ret = cublasSetVector(ma->dimension(), sizeof(float), ma->array, 1, da, 1);
    ret = cublasSetVector(mb->dimension(), sizeof(float), mb->array, 1, db, 1);
    matrixDiv(da, db, dc, mc->dimension());
    ret = cublasGetVector(mc->dimension(), sizeof(float), dc, 1, mc->array, 1);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

void matrixDiv(Matrix<double> *ma, Matrix<double> *mb, Matrix<double> *mc)
{
    CHECK(ma->dimension() == mb->dimension());
    CHECK(mb->dimension() == mc->dimension());
    double *da, *db, *dc;
    int ret = -1;
    cudaMalloc((void **)&da, ma->dimension() * sizeof(double));
    cudaMalloc((void **)&db, mb->dimension() * sizeof(double));
    cudaMalloc((void **)&dc, mc->dimension() * sizeof(double));
    ret = cublasSetVector(ma->dimension(), sizeof(double), ma->array, 1, da, 1);
    ret = cublasSetVector(mb->dimension(), sizeof(double), mb->array, 1, db, 1);
    matrixDiv(da, db, dc, mc->dimension());
    ret = cublasGetVector(mc->dimension(), sizeof(double), dc, 1, mc->array, 1);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

// float, C = A * B, A(m*k), B(k*n), C(m*n);
void matrixGemm(float *A, float *B, float *C, int m, int n, int k)
{
    float alpha = 1.0, beta = 0.0;
    cublasHandle_t handle;
    int ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, 
                          &alpha, B, n, A, k, &beta, C, n);
    if(ret != CUBLAS_STATUS_SUCCESS)
    {
        CUDA_OPERATION_ERROR(ret);
    }
    ret = cublasDestroy(handle);
}

// double, C = A * B
void matrixGemm(double *A, double *B, double *C, int m, int n, int k)
{
    double alpha = 1.0, beta = 0.0;
    cublasHandle_t handle;
    int ret = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, 
                          &alpha, B, n, A, k, &beta, C, n);
    if(ret != CUBLAS_STATUS_SUCCESS)
    {
        CUDA_OPERATION_ERROR(ret);
    }
    ret = cublasDestroy(handle);
}

void matrixGemm(Matrix<float> *ma, Matrix<float> *mb, Matrix<float> *mc)
{
    CHECK(ma->vsize[1] == mb->vsize[0]);
    CHECK(mc->vsize[0] == ma->vsize[0]);
    CHECK(mc->vsize[1] == mb->vsize[1]);
    float *da, *db, *dc;
    int ret = -1;
    cudaMalloc((void **)&da, ma->dimension() * sizeof(float));
    cudaMalloc((void **)&db, mb->dimension() * sizeof(float));
    cudaMalloc((void **)&dc, mc->dimension() * sizeof(float));
    ret = cublasSetVector(ma->dimension(), sizeof(float), ma->array, 1, da, 1);
    ret = cublasSetVector(mb->dimension(), sizeof(float), mb->array, 1, db, 1);
    matrixGemm(da, db, dc, ma->vsize[0], mb->vsize[1], ma->vsize[1]);
    ret = cublasGetVector(mc->dimension(), sizeof(float), dc, 1, mc->array, 1);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

void matrixGemm(Matrix<double> *ma, Matrix<double> *mb, Matrix<double> *mc)
{
    CHECK(ma->vsize[1] == mb->vsize[0]);
    CHECK(mc->vsize[0] == ma->vsize[0]);
    CHECK(mc->vsize[1] == mb->vsize[1]);
    double *da, *db, *dc;
    int ret = -1;
    cudaMalloc((void **)&da, ma->dimension() * sizeof(double));
    cudaMalloc((void **)&db, mb->dimension() * sizeof(double));
    cudaMalloc((void **)&dc, mc->dimension() * sizeof(double));
    ret = cublasSetVector(ma->dimension(), sizeof(double), ma->array, 1, da, 1);
    ret = cublasSetVector(mb->dimension(), sizeof(double), mb->array, 1, db, 1);
    matrixGemm(da, db, dc, ma->vsize[0], mb->vsize[1], ma->vsize[1]);
    ret = cublasGetVector(mc->dimension(), sizeof(double), dc, 1, mc->array, 1);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

void matrixGemmTest()
{

}

// float, C = A_T * B
void matrixTransGemm(float *A, float *B, float *C, int m, int n, int k)
{
    float alpha = 1.0, beta = 0.0;
    cublasHandle_t handle;
    int ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, 
                          &alpha, B, n, A, m, &beta, C, n);
    if(ret != CUBLAS_STATUS_SUCCESS)
    {
        CUDA_OPERATION_ERROR(ret);
    }
    ret = cublasDestroy(handle);
}

// double, C = A_T * B
void matrixTransGemm(double *A, double *B, double *C, int m, int n, int k)
{
    double alpha = 1.0, beta = 0.0;
    cublasHandle_t handle;
    int ret = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, 
                          &alpha, B, n, A, m, &beta, C, n);
    ret = cublasDestroy(handle);
}



