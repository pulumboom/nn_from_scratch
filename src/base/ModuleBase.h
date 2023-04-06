#pragma once

#include "Eigen/Dense"

class ModuleBase {
public:
    Eigen::MatrixXd operator()(Eigen::MatrixXd &input) {
        return Forward(input);
    }

    virtual Eigen::MatrixXd Forward(const Eigen::MatrixXd &input) = 0;

    virtual Eigen::MatrixXd Backward(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) = 0;

    virtual void ResetGrad() = 0;

    virtual void UpdateParameters(const Eigen::MatrixXd &input, Eigen::MatrixXd &grad_output) = 0;

    void Train() {
        training_mode_ = true;
    }

    void Eval() {
        training_mode_ = false;
    }

    const Eigen::MatrixXd &Output() const {
        return output_;
    }

protected:
    Eigen::MatrixXd output_;
    bool training_mode_ = true;
};