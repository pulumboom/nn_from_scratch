#include "Sigmoid.h"

Eigen::MatrixXd Sigmoid::Forward(const Eigen::MatrixXd &input) {
    output_ = 1 / (1 + input.array().exp());
    return output_;
}

Eigen::MatrixXd Sigmoid::Backward(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) {
    output_ = Forward(input);
    return grad_output.array() * output_.array() * (1 - output_.array());
}

void Sigmoid::ResetGrad() {}

void Sigmoid::UpdateParameters(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) {}
