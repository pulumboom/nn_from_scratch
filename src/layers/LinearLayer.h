#pragma once

#include "Eigen/Dense"

#include "ModuleBase.h"

class LinearLayer : public ModuleBase {
public:
    LinearLayer(long long in_features, long long out_features, bool bias=true);

    Eigen::MatrixXd Forward(const Eigen::MatrixXd &input) override;
    Eigen::MatrixXd Backward(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) override;

    void ResetGrad() override;
    void UpdateParameters(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) override;

    Eigen::MatrixXd Weights();
    Eigen::MatrixXd Bias();
private:
    Eigen::MatrixXd weights_; // in_features x out_features
    Eigen::MatrixXd grad_for_weights_; // in_features x out_features
    bool has_bias_;
    Eigen::VectorXd bias_; // out_features x 1
    Eigen::VectorXd grad_for_bias_; // out_features x 1
};
