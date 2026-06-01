#include <unn/Softmax.hpp>

namespace unn
{
Eigen::MatrixXd Softmax::operator()(const Eigen::MatrixXd &inputs)
{
  // Row vector containing the maximum input of each sample
  Eigen::RowVectorXd maxes = inputs.colwise().maxCoeff();

  // Adjust inputs
  Eigen::MatrixXd adjusted = inputs.rowwise() - maxes;

  // Apply exponential
  adjusted = adjusted.array().exp();

  // Get sum of each sample
  Eigen::RowVectorXd s = adjusted.colwise().sum();

  // Softmax (hopefully)
  m_out = adjusted.array().rowwise() / s.array();

  return m_out;
}
void Softmax::backward(const Eigen::MatrixXd &d_next)
{
  // BEGIN ASSERTIONS
  const bool valid_d_next_shape = d_next.rows() == m_out.rows() &&
                                  d_next.cols() == m_out.cols();
  assert(((valid_d_next_shape) && "d_next has invalid shape"));
  // END ASSERTIONS

  m_d_inputs.resize(m_out.rows(), m_out.cols());

  for (Eigen::Index j = 0; j < m_out.cols(); j++) {
    Eigen::MatrixXd jacobian = static_cast<Eigen::MatrixXd>(m_out.col(j).asDiagonal()) -
                               (m_out.col(j) * m_out.col(j).transpose());
    m_d_inputs.col(j) = jacobian * d_next.col(j);
  }
}
} // namespace unn
