#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Core>

namespace unn
{
struct Layer {
  virtual void backward(const Eigen::MatrixXd &d_next);
};
} // namespace unn

#endif
