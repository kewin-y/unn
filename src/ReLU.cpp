#include <unn/ReLU.hpp>

namespace unn
{
Eigen::MatrixXd ReLU::operator()(const Eigen::MatrixXd &inputs)
{
  in = inputs;
  return inputs.cwiseMax(0);
}
void ReLU::backward(const Eigen::MatrixXd &d_next)
{

  const bool valid_d_next_rows = d_next.rows() == in.rows();
  const bool valid_d_next_cols = d_next.cols() == in.cols();

  assert(((valid_d_next_rows && valid_d_next_cols) && "d_next has invalid shape"));

  d_in = d_next;
  // https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1DenseBase.html#ae565bdc9af5d0d7d6f86eeceefa0041f
  d_in = (in.array() > 0.0).select(d_in, 0.0);
}
} // namespace unn
