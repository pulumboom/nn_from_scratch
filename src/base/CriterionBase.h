#pragma once

#include "Eigen/Dense"

class CriterionBase {
public:
    double operator()(Eigen::MatrixXd &input, Eigen::MatrixXd &target) {
        return Forward(input, target);
    }

    virtual double Forward(Eigen::MatrixXd &input, Eigen::MatrixXd &target) = 0;
    virtual Eigen::MatrixXd Backward(Eigen::MatrixXd &input, Eigen::MatrixXd &target) = 0;

protected:
    double output_ = 0;
};
