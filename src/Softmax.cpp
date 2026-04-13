#include <unn/Softmax.hpp>

namespace unn
{
Eigen::MatrixXd Softmax::operator()(const Eigen::MatrixXd &inputs) const
{
  // Column vector containing the maximum input of each sample
  Eigen::RowVectorXd maxes = inputs.colwise().maxCoeff();

  // Adjust inputs
  Eigen::MatrixXd adjusted = inputs.rowwise() - maxes;

  // Apply exponential
  adjusted = adjusted.array().exp();

  // Get sum of each sample
  Eigen::RowVectorXd s = adjusted.colwise().sum();

  // Softmax (hopefully)
  return adjusted.array().rowwise() / s.array();
}
} // namespace unn
