#pragma once

#include "../base/CriterionBase.h"

class CrossEntropyLoss : public CriterionBase {
public:
    double Forward(Eigen::MatrixXd &input, Eigen::MatrixXd &target) override;
    Eigen::MatrixXd Backward(Eigen::MatrixXd &input, Eigen::MatrixXd &target) override;

private:
//todo
};

