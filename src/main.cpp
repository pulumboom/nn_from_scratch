//#include "nn/MLP.h"

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <any>

#include "LinearLayer.h"
#include "ReLU.h"
#include "Sequential.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

template<typename T>
int foo(T x) {
    int y = x;
    y++;
    return y;
}

int main() {
    std::vector<ModuleTypeErasure> mm = {LinearLayer(1, 2), ReLU()};
}
