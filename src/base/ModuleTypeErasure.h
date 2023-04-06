#pragma once

#include <memory>
#include <iostream> // todo

#include "Eigen/Dense"

#include "ModuleBase.h"

class ModuleTypeErasure {
public:
    template<typename Layer>
    ModuleTypeErasure(Layer layer) : p_(
            std::make_unique<Layer>(std::move(layer))
            ) {}

    ModuleTypeErasure(ModuleTypeErasure&&) noexcept = default;
    ModuleTypeErasure& operator=(ModuleTypeErasure&&) noexcept = default;

    Eigen::MatrixXd operator()(const Eigen::MatrixXd &input) {
        return Forward(input);
    }

    Eigen::MatrixXd Forward(const Eigen::MatrixXd &input) {
        return p_->Forward(input);
    }

    Eigen::MatrixXd Backward(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) {
        return p_->Backward(input, grad_output);
    }

    void ResetGrad() {
        p_->ResetGrad();
    }

    void UpdateParameters(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) {
        p_->UpdateParameters(input, grad_output);
    }

    const Eigen::MatrixXd& Output() {
        return p_->Output();
    }

private:
    std::unique_ptr<ModuleBase> p_ = nullptr;
};