#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Core>

namespace unn
{
struct Layer {
  virtual void backward(const Eigen::MatrixXd &d_next) = 0;
  virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) = 0;
  virtual ~Layer() = default;
};
} // namespace unn

#endif
