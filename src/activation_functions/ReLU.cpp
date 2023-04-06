#include "ReLU.h"

Eigen::MatrixXd ReLU::Forward(const Eigen::MatrixXd &input) {
    return input.cwiseMax(0);
}

Eigen::MatrixXd ReLU::Backward(const Eigen::MatrixXd &input, Eigen::MatrixXd& grad_output) {
    return (input.array() > 0).select(grad_output, 0.0);
}

void ReLU::ResetGrad() {}

void ReLU::UpdateParameters(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) {}
