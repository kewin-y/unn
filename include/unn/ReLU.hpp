#ifndef UNN_RELU_HPP
#define UNN_RELU_HPP

#include <Eigen/Core>

namespace unn
{
struct ReLU {
  ReLU() = default;
  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) const;
};
} // namespace unn

#endif // UNN_RELU_HPP
