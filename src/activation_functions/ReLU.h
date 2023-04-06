#pragma once

#include "ModuleBase.h"

class ReLU : public ModuleBase {
public:
    Eigen::MatrixXd Forward(const Eigen::MatrixXd &input) override;
    Eigen::MatrixXd Backward(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) override;

    void ResetGrad() override;
    void UpdateParameters(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) override;
};
