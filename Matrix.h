#ifndef _MATRIX_H
#define _MATRIX_H

#include <vector>
#include <random>
#include <typeinfo>

#include <string.h>

#include "util/Util.h"
#include "util/Types.h"

enum
{
    M_FULL,
    M_RAND,
    M_EYE,
    M_DIAG
};

template<typename Dtype>
class Matrix
{
    public:
    Dtype *array;             // storage 
    uint64_t dim;             // dimension
    std::vector<int> vsize;   // size
    std::vector<int> vstride; // stride
    
    public:

    // fast set of 1-dim matrix
    Matrix(int d)
    {
        array = new Dtype[d];
        dim = d;
        vsize.push_back(d);
        vstride.push_back(1);
    }

    // fast set of 2-dim matrix
    Matrix(int m, int n)
    {
        dim = m * n;
        vsize.push_back(m);
        vsize.push_back(n);
        array = new Dtype[dim];
        vstride.push_back(n);
        vstride.push_back(1);
    }

    Matrix(std::vector<int> sz)
    {
        dim = getDim(sz);
        vsize = sz;
        array = new Dtype[dim];
    }

    Matrix(int d, std::vector<int> sz)
    {
        dim = d;
        vsize = sz;
        array = new Dtype[dim];
    }

    Matrix(int d, Dtype start, Dtype end, int type)
    {
        dim = d;
        vsize.push_back(d);
        vstride.push_back(1);
        array = new Dtype[dim];
        std::default_random_engine e;
        std::uniform_real_distribution<Dtype> ud(start, end);
        for(int i = 0; i < dim; i++)
        {
            array[i] = (type == M_FULL ? start : ud(e));
        }
    }

    // fast set of random 2-dim matrix
    // if type is full, set start only
    Matrix(int m, int n, Dtype start, Dtype end, int type)
    {
        dim = m * n;
        vsize.push_back(m);
        vsize.push_back(n);
        array = new Dtype[dim];
        vstride.push_back(n);
        vstride.push_back(1);
        std::default_random_engine e;
        std::uniform_real_distribution<Dtype> ud(start, end);
        for(int i = 0; i < dim; i++)
        {
            array[i] = (type == M_FULL ? start : ud(e));
        }
    }

    Matrix(int m, int n, int type = M_EYE)
    {
        dim = m * n;
        vsize.push_back(m);
        vsize.push_back(n);
        array = new Dtype[dim];
        vstride.push_back(n);
        vstride.push_back(1);

    }

    Matrix(std::vector<int> sz, Dtype start, Dtype end, int type)
    {
        dim = getDim(sz);
        vsize = sz;
        // TODO: stide
        // in fact, i don't care stride here
        array = new Dtype[dim];
        std::default_random_engine e;
        std::uniform_real_distribution<Dtype> ud(start, end);
        for(int i = 0; i < dim; i++)
        {
            array[i] = (type == M_FULL ? start : ud(e));
        }
    }

    int getDim(std::vector<int> sz)
    {
        int res = 1;
        for(int i = 0; i < sz.size(); i++)
        {
            res *= 1;
        }
        return res;
    }

    int dimension()
    {
        if(dim != -1)
        {
            return dim;
        }

        dim = getDim(vsize);
        return dim;
    }

    void reshape(std::vector<int> sz)
    {
        CHECK(dim == getDim(sz));
        vsize = sz;
    }

    void clear()
    {
        dim = -1;
        vsize.clear();
        vstride.clear();
    }

    void set0()
    {
        memset(array, 0, dim * sizeof(Dtype));
    }

    bool checkNull()
    {
        return dim == -1 ? true : false;
    }

    void print()
    {
        for(int i = 0; i < dim; i++)
        {
            printf("%lf ", this->array[i]);
            if((i + 1) % this->vsize[0] == 0)
            {
                printf("\n");
            }
        }
        printf("\n");
    }

    Dtype & operator[](int i)
    {
        return array[i];
    }

    Dtype & element(int i, int j)
    {
        return array[i * vsize[1] + j];
    }

    Matrix<Dtype> operator+(const Matrix<Dtype> m)
    {
        Matrix<Dtype> res(dimension(), vsize);
        for(int i = 0; i < dimension(); i++)
        {
            res.array[i] = array[i] + m.array[i];
        }
        return res;
    }

    Matrix<Dtype> operator-(const Matrix<Dtype> m)
    {
        Matrix<Dtype> res(dimension(), vsize);
        for(int i = 0; i < dimension(); i++)
        {
            res.array[i] = array[i] - m.array[i];
        }
        return res;
    }

    // assume matrix is 1-dim or 2-dim
    Matrix<Dtype> operator*(Matrix<Dtype> m)
    {
        Matrix<Dtype> res(vsize[0], m.vsize[1]);
        // TODO: mkl here
        // naive implement, for test gpu
        for(int i = 0; i < vsize[0]; i++)
        {
            for(int j = 0; j < m.vsize[1]; j++)
            {
                res.element(i, j) = 0;
                for(int k = 0; k < vsize[1]; k++)
                {
                    res.element(i, j) += element(i, k) * m.element(k, j);
                }
            }
        }
        return res;
    }

    Matrix<Dtype> operator/(const Dtype d)
    {
        Matrix<Dtype> res(dimension(), vsize);
        for(int i = 0; i < dimension(); i++)
        {
            res.array[i] = array[i] / d;
        }
        return res;
    }
};

// TODO: CPU
// gemm, convIm2col, convFFT, convWinograd
// template<typename Dtype>
// void cMatrixConvFFT(Matrix<Dtype> image, Matrix<Dtype> kernel, int stride);

// template<typename Dtype>
// void cMatrixConvIm2col(Matrix<Dtype> image, Matrix<Dtype> kernel, int stride);

// template<typename Dtype>
// void cMatrixConvWinograd(Matrix<Dtype> image, Matrix<Dtype> kernel, int stride);

// GPU
#define DEFAULT_THREAD_NUM 512
#define GET_BLOCK_NUM(n) (((n) + DEFAULT_THREAD_NUM - 1) / DEFAULT_THREAD_NUM)


// template<typename Dtype>
// void gMatrixConvFFT(Matrix<Dtype> image, Matrix<Dtype> kernel, int stride);

// template<typename Dtype>
// void gMatrixConvIm2col(Matrix<Dtype> image, Matrix<Dtype> kernel, int stride);

// template<typename Dtype>
// void gMatrixConvWinograd(Matrix<Dtype> image, Matrix<Dtype> kernel, int stride);


#endif // _MATRIX_H