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
  out = adjusted.array().rowwise() / s.array();

  return out;
}
void Softmax::backward(const Eigen::MatrixXd &d_next)
{
  // ========== BEGIN ASSERTIONS ==========
  const bool valid_d_next_shape = d_next.rows() == out.rows() &&
                                  d_next.cols() == out.cols();
  assert(((valid_d_next_shape) && "d_next has invalid shape"));
  // ========== END ASSERTIONS ==========

  d_in.resize(out.rows(), out.cols());

  for (Eigen::Index j = 0; j < out.cols(); j++) {
    Eigen::MatrixXd jacobian = static_cast<Eigen::MatrixXd>(out.col(j).asDiagonal()) -
                               (out.col(j) * out.col(j).transpose());
    d_in.col(j) = jacobian * out.col(j);
  }
}
} // namespace unn
