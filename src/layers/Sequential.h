#pragma once

#include <vector>
#include <any>

#include "Eigen/Dense"

#include "ModuleTypeErasure.h"

class Sequential : public ModuleBase {
public:
    Sequential(std::vector<ModuleTypeErasure> layers);

    Eigen::MatrixXd Forward(const Eigen::MatrixXd &input) override;
    Eigen::MatrixXd Backward(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) override;

    void ResetGrad() override;
    void UpdateParameters(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) override;
private:
    std::vector<ModuleTypeErasure> layers_;
};