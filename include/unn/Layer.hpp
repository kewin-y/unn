#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Core>

namespace unn
{
class Layer
{
protected:
  Eigen::MatrixXd input;
  Eigen::MatrixXd d_input;
public:
  virtual void backward(const Eigen::MatrixXd &d_next) = 0;
  virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &input) = 0;
};
} // namespace unn

#endif
