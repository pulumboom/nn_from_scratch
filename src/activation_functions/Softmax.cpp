#include "Softmax.h"

Eigen::MatrixXd Softmax::Forward(const Eigen::MatrixXd &input) {
    output_ = input.array().exp();
    Eigen::VectorXd column_sum = output_.colwise().sum();
    output_.array().rowwise() /= column_sum.array().transpose();
    return output_;
}

Eigen::MatrixXd Softmax::Backward(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) {
    output_ = Forward(input);
//    dinput = -output_.rowwise().transpose() * output_.array()
    // todo
}

void Softmax::ResetGrad() {}

void Softmax::UpdateParameters(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) {}
