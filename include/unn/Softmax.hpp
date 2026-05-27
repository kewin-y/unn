#ifndef UNN_SOFTMAX_HPP
#define UNN_SOFTMAX_HPP

#include "unn/Layer.hpp"
#include <Eigen/Core>

namespace unn
{
struct Softmax : Layer {
  Softmax() = default;

  // shape(inputs) = (features, sample)
  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) override;
  void backward(const Eigen::MatrixXd &d_next) override;

private:

  // FORWARD PASS
  Eigen::MatrixXd in;

  // BACKWARD PASS
  Eigen::MatrixXd d_in;

};
} // namespace unn

#endif // UNN_SOFTMAX_HPP
