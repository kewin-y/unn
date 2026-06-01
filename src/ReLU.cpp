#include <unn/ReLU.hpp>

namespace unn
{
Eigen::MatrixXd ReLU::operator()(const Eigen::MatrixXd &inputs)
{
  m_inputs = inputs;
  return inputs.cwiseMax(0);
}
void ReLU::backward(const Eigen::MatrixXd &d_next)
{

  const bool valid_d_next_rows = d_next.rows() == m_inputs.rows();
  const bool valid_d_next_cols = d_next.cols() == m_inputs.cols();

  assert(((valid_d_next_rows && valid_d_next_cols) && "d_next has invalid shape"));

  m_d_inputs = d_next;

  // https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1DenseBase.html#ae565bdc9af5d0d7d6f86eeceefa0041f
  m_d_inputs = (m_inputs.array() > 0.0).select(m_d_inputs, 0.0);
}
} // namespace unn
