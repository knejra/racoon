#include "CGraph.h"

template<typename Dtype>
void CGraph<Dtype>::toposort(int start, int end)
{

}

template<typename Dtype>
void CGraph<Dtype>::forward(int idx)
{
    for(int i = 0; i < nodes[idx].priors.size(); i++)
    {
        if(nodes[nodes[idx].priors[i]].value.checkNull())
        {
            forward(nodes[idx].priors[i]);
        }
    }   
    nodes[idx].compute();
}

template<typename Dtype>
void CGraph<Dtype>::autograd(Matrix<Dtype> grad, int start, int end)
{
    nodeToGrad[end].push_back(grad);
    std::vector<int> topos = toposort(start, end);
    for(int i = 0; i < topos.size(); i++)
    {
        if(!nodeToGrad.count(topos[i]))
        {
            continue;
        }

        nodes[topos[i]].gradient();
        nodes[topos[i]].partialGradient();
    }
}