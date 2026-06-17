#include "unn/Loss_CCE.hpp"
#include <Eigen/Core>

/**
 * Clamps each entry in `mat` between 1e-7 and 1 - 1e-7
 * and takes the natural logarithm of each entry
 */

namespace
{
Eigen::MatrixXd clip_log(const Eigen::MatrixXd &mat)
{
  return mat.cwiseMax(1e-7).cwiseMin(1 - 1e-7).array().log();
}
} // namespace

namespace unn
{

Eigen::MatrixXd Loss_CCE::operator()(const Eigen::MatrixXd &input, const Eigen::RowVectorXi &target)
{
  // ASSERTIONS
  const bool valid_input_range = target.minCoeff() >= 0 && target.maxCoeff() < input.rows();
  assert(((valid_input_range) && "Invalid range of values exist in y. Make sure 0 <= y < input.rows()."));
  // END ASSERTIONS

  // Calculate Loss
  const auto samples = input.cols();
  const auto classes = input.rows();
  const auto input_clip_log = clip_log(input);

  Eigen::MatrixXd losses(1, samples);

  for (Eigen::Index i = 0; i < samples; i++) {
    losses(0, i) = -input_clip_log(target(i), i);
  }

  // Calculate target as a one hot matrix
  Eigen::MatrixXd one_hot(classes, samples);
  one_hot.setZero();

  for (Eigen::Index i = 0; i < samples; i++) {
    one_hot(target(i), i) = 1;
  }

  // Update member variables
  this->average_loss = losses.mean();
  this->target = std::move(one_hot);
  this->input = input;

  return losses;
}

Eigen::MatrixXd Loss_CCE::operator()(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target)
{
  const auto input_clip_log = clip_log(input);

  // This code opts for element-wise operations.
  // Doing matrix product in: -(y_true * y_pred_clip_log).diagonal();
  // wastes a lot of computation b/c non-diagonal
  // entries are calculated for no reason
  Eigen::MatrixXd losses = -(target.array() * input_clip_log.array()).colwise().sum();

  // Update member variables
  this->average_loss = losses.mean();
  this->input = input;
  this->target = target;

  return losses;
}

void Loss_CCE::backward()
{
  // Divide each entry by n_samples to normalize gradient
  const auto samples = input.cols();
  this->d_input = -(target.array() / input.array()) / samples;
};
} // namespace unn
