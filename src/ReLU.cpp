#include <unn/ReLU.hpp>

namespace unn
{
Eigen::MatrixXd ReLU::operator()(const Eigen::MatrixXd &input)
{
  this->input = input;
  return input.cwiseMax(0);
}
void ReLU::backward(const Eigen::MatrixXd &d_next)
{
  // https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1DenseBase.html#ae565bdc9af5d0d7d6f86eeceefa0041f
  this->d_input = (input.array() > 0.0).select(d_next, 0.0);
}
} // namespace unn
