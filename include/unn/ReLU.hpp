#ifndef UNN_RELU_HPP
#define UNN_RELU_HPP

#include "unn/Layer.hpp"
#include <Eigen/Core>

namespace unn
{
class ReLU : public Layer
{
public:
  ReLU() = default;

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) override;
  void backward(const Eigen::MatrixXd &d_next) override;
};
} // namespace unn

#endif // UNN_RELU_HPP
