#include "Sequential.h"

Sequential::Sequential(std::vector<ModuleTypeErasure> layers) : layers_(std::move(layers)) {}

Eigen::MatrixXd Sequential::Forward(const Eigen::MatrixXd &input) {
    output_ = input;
    for (auto &layer : layers_) {
        output_ = layer.Forward(input);
    }
    return output_;
}

Eigen::MatrixXd Sequential::Backward(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) {
    output_ = Forward(input);
    for (int i = layers_.size() - 1; i > -1; --i) {
        if (i > 0) {
            layers_[i].UpdateParameters(layers_[i - 1].Output(), grad_output);
            grad_output = layers_[i].Backward(layers_[i - 1].Output(), grad_output);
        } else {
            layers_[i].UpdateParameters(input, grad_output);
            grad_output = layers_[i].Backward(input, grad_output);
        }
    }
    return grad_output;
}

void Sequential::ResetGrad() {
    for (auto &layer : layers_) {
        layer.ResetGrad();
    }
}

void Sequential::UpdateParameters(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) {}
