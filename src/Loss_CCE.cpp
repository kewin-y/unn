#include <unn/Loss_CCE.hpp>

#include <cassert>

// Clamps each entry in `mat` between 1e-7 and 1 - 1e-7
// and takes the natural logarithm of each entry
Eigen::MatrixXd clip_log(const Eigen::MatrixXd &mat)
{
  return mat.cwiseMax(1e-7).cwiseMin(1 - 1e-7).array().log();
}

namespace unn
{

Eigen::VectorXd Loss_CCE::operator()(const Eigen::MatrixXd &y_pred, const Eigen::VectorXi &y) const
{
  const bool valid_y_pred_rows = y_pred.rows() == y.rows();
  assert(((valid_y_pred_rows) && "mismatch in size between y_pred and y"));

  const bool valid_y_range = y.minCoeff() >= 0 && y.maxCoeff() < y.rows();
  assert(((valid_y_range) && "invalid range of values exist in y"));

  const auto y_pred_clip_log = clip_log(y_pred);

  Eigen::VectorXd neg_result(y.rows());

  for (Eigen::Index i = 0; i < neg_result.rows(); ++i) {
    neg_result(i) = y_pred_clip_log(y(i), i);
  }

  // Negation of Negated Result -> Result
  return -neg_result;
}

Eigen::VectorXd Loss_CCE::operator()(const Eigen::MatrixXd &y_pred, const Eigen::MatrixXd &y) const
{
  const bool valid_size = y_pred.rows() == y.rows() && y_pred.cols() == y.cols();
  assert(((valid_size) && "mismatch in size between y_pred and y"));

  const auto y_pred_clip_log = clip_log(y_pred);

  return -(y * y_pred_clip_log).diagonal();
}
} // namespace unn
