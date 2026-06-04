#ifndef UNN_SOFTMAX_HPP
#define UNN_SOFTMAX_HPP

#include "unn/Layer.hpp"
#include <Eigen/Core>

namespace unn
{
struct Softmax : Layer {
  Softmax() = default;

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) override;
  void backward(const Eigen::MatrixXd &d_next) override;

  // FORWARD PASS
  Eigen::MatrixXd out;

  // BACKWARD PASS
  Eigen::MatrixXd d_inputs;

};
} // namespace unn

#endif // UNN_SOFTMAX_HPP
