#ifndef _CGRAPH_H
#define _CGRAPH_H

#include <vector>
#include <map>
#include <memory>
#include <cmath>

#include "Matrix.h"

#include "util/Util.h"

#define tanh(x) ((1 - exp(-(x))) / (1 + exp(-(x))))
#define sigmoid(x) (1 / (1 + exp(-(x))))

enum
{
    D_CPU,
    D_GPU
};

template<typename Dtype>
class Node
{
    public:
    int nodeId;                                   // node identifier
    int devType;                                  // device type: cpu or gpu
    Matrix<Dtype> value;                          // node's value
    Matrix<Dtype> gradient;                       // node's gradient
    std::vector<int> priors;                      // prior nodes
    std::vector<int> posts;                       // post nodes
    std::shared_ptr<CGraph<Dtype>> defaultGraph;  // graph

    public:

    // compute: virtual, compute this node's value
    //          parameters from priors
    virtual void compute();

    // jacobi: virtual, compute this node's jacobi
    //         dz/dp, while z means current node,
    //         p means prior node
    virtual Matrix<Dtype> jacobi();

    // partialGradient: compute this node to prior nodes's
    //                  partial gradient;
    void partialGradient()
    {
        // Matrix<Dtype> res = (devType == D_CPU) ? gradient * jacobi() : gpuMatrixGemm(gradient, jacobi());
        Matrix<Dtype> res = gradient * jacobi();
        for(int i = 0; i < priors.size(); i++)
        {
            defaultGraph->nodeToGrad[priors[i]].push_back(res);
        }
        return res;
    }

    // gradient: compute gradient on this node
    // parameters: backgrad-backpassed gradient
    // outputs   : void
    void gradient()
    {
        gradient.set0();
        for(int i = 0; i < defaultGraph->nodeToGrad[nodeId].size(); i++)
        {
            gradient += defaultGraph->nodeToGrad[nodeId][i];
        }
    }
};

template<typename Dtype>
class Variable : public Node<Dtype>
{
    
};

template<typename Dtype>
class Add : public Node<Dtype>
{
    public:
    void compute()
    {
        this->value = this->defaultGraph->nodes[this->priors[0]].value + \
                      this->defaultGraph->nodes[this->priors[1]].value;  
    }

    Matrix<Dtype> jacobi()
    {
        Matrix<Dtype> res(this->value.vsize[0], this->value.vsize[1], M_EYE);
        return res;
    }
};

template<typename Dtype>
class Sub : public Node<Dtype>
{

};

template<typename Dtype>
class Mul : public Node<Dtype>
{
    public:

    void compute()
    {
        this->value = this->defaultGraph->nodes[this->priors[0]].value * \
                      this->defaultGraph->nodes[this->priors[1]].value;
    }

    Matrix<Dtype> jacobi()
    {
        Matrix<Dtype> res;
        return res;
    }
};

template<typename Dtype>
class Div : public Node<Dtype>
{

};

template<typename Dtype>
class Relu : public Node<Dtype>
{
    public:

    void compute()
    {
        for(int i = 0; i < this->value.dimension(); i++)
        {
            this->value.array[i] = this->defaultGraph[this->priors[0]].value[i] > 0 ? \
                                   this->defaultGraph[this->priors[0]].value[i] : 0;
        }
    }

    Matrix<Dtype> jacobi()
    {
        Matrix<Dtype> res = this->value;
        for(int i = 0; i < res.dimension(); i++)
        {
            res.array[i] = res.array[i] > 0 ? 1 : 0;
        }
        return res;
    }
};

template<typename Dtype>
class Sigmoid : public Node<Dtype>
{
    public:

    void compute()
    {
        for(int i = 0; i < this->value.dimension(); i++)
        {
            this->value[i] = sigmoid(this->defaultGraph->nodes[this->priors[0]].value[i]);
        }
    }

    Matrix<Dtype> jacobi()
    {
        return this->value * (1 - this->value);
    }
};

template<typename Dtype>
class Tanh : public Node<Dtype>
{
    public:
    void compute()
    {
        for(int i = 0; i < this->value.dimension(); i++)
        {
            this->value[i] = tanh(this->defaultGraph->nodes[this->priors[0]].value[i]);
        }
    }

    Matrix<Dtype> jacobi()
    {
        return 1 - this->value * this->value;
    }
};

template<typename Dtype>
class CrossEntropy : public Node<Dtype>
{

};

template<typename Dtype>
class Softmax : public Node<Dtype>
{

};

template<typename Dtype>
class Convlution : public Node<Dtype>
{

};

template<typename Dtype>
class Adam : public Node<Dtype>
{

};

template<typename Dtype>
class CGraph
{
    public:
    std::map<int, std::vector<Matrix<Dtype>>> nodeToGrad;
    std::vector<Node<Dtype>> nodes;

    public:

    // initialize: set opMap and clear nodes
    // parameters: void
    // outpus    : void
    void initialize();

    // toposort: toposort on computation graph
    //           for autograd(autodiff)
    // parameters: start-start node index
    //             end-end node index
    // outputs   : toposort sequence of nodes
    std::vector<int> toposort(int start, int end);

    // forward: forward pass on network
    // parameters: idx-node's index
    // outputs   : void
    void forward(int idx);

    // autograd: autograd backward pass on network
    // parameters: grad-gradient of current node
    //             startIdx-start node index
    //             endIdx-end nodex index
    // outputs   : void
    void autograd(Matrix<Dtype> grad, int start, int end);
};

#endif // _CGRAPH_H