#include "MSE.h"

double MSE::Forward(Eigen::MatrixXd &input, Eigen::MatrixXd &target) {
    output_ = (input - target).array().pow(2).mean();
    return output_;
}

Eigen::MatrixXd MSE::Backward(Eigen::MatrixXd &input, Eigen::MatrixXd &target) {
    return 2 * (input - target);
}
