#include "LinearLayer.h"


LinearLayer::LinearLayer(long long int in_features, long long int out_features, bool bias) : has_bias_(bias) {
    weights_ = Eigen::MatrixXd::Random(in_features, out_features);
    grad_for_weights_ = Eigen::MatrixXd::Zero(in_features, out_features);

    if (bias) {
        bias_ = Eigen::VectorXd::Random(out_features);
        grad_for_bias_ = Eigen::VectorXd::Zero(out_features);
    }
}

Eigen::MatrixXd LinearLayer::Forward(const Eigen::MatrixXd &input) {
    output_ = (input * weights_.transpose()).rowwise() + bias_.transpose();
    return output_;
}

Eigen::MatrixXd LinearLayer::Backward(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) {
    Forward(input);
    return grad_output * weights_;
}

void LinearLayer::ResetGrad() {
    grad_for_weights_.setZero();

    if (has_bias_) {
        grad_for_bias_.setZero();
    }
}

void LinearLayer::UpdateParameters(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) {
    grad_for_weights_ += grad_output.transpose() * input;

    if (has_bias_) {
        grad_for_bias_ += grad_output.colwise().sum();
    }
}

Eigen::MatrixXd LinearLayer::Weights() {
    return weights_;
}

Eigen::MatrixXd LinearLayer::Bias() {
    return bias_;
}
