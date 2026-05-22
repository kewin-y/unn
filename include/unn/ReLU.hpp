#ifndef UNN_RELU_HPP
#define UNN_RELU_HPP

#include "unn/Layer.hpp"
#include <Eigen/Core>

namespace unn
{
struct ReLU : Layer {
  ReLU() = default;

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) override;
  void backward(const Eigen::MatrixXd &d_next) override;

private:
  // Forward Pass
  Eigen::MatrixXd in;

  // Backward Pass
  Eigen::MatrixXd d_in;
};
} // namespace unn

#endif // UNN_RELU_HPP
