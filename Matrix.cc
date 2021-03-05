#include "Matrix.h"

void cpuMatrixTest()
{
    Matrix<float> a(5, 3, 1.0, 9.0, M_RAND);
    a.print();
    Matrix<float> b(5, 3, 2.0, 10.0, M_RAND);
    b.print();
    Matrix<float> mAdd = a + b;
    mAdd.print();
    Matrix<float> mSub = a - b;
    mSub.print();
    Matrix<float> c(3, 4, 1.0, 9.0, M_RAND);
    Matrix<float> mMul = a * c;
    mMul.print();
    Matrix<float> mDiv = a / 2.0;
    mDiv.print();
}

int main()
{
    cpuMatrixTest();
}