#ifndef UNN_LOSS_CCE_HPP
#define UNN_LOSS_CCE_HPP

#include "unn/Layer.hpp"
#include <Eigen/Core>
#include <type_traits>

/**
 * Clamps each entry in `mat` between 1e-7 and 1 - 1e-7
 * and takes the natural logarithm of each entry
 */
inline Eigen::MatrixXd clip_log(const Eigen::MatrixXd &mat)
{
  return mat.cwiseMax(1e-7).cwiseMin(1 - 1e-7).array().log();
}

namespace unn
{
template <typename TargetType>
struct Loss_CCE : Layer {
  Loss_CCE(const TargetType &targets) : y_true(targets) {};

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &predictions) override;
  void backward(const Eigen::MatrixXd &d_next) override;

private:
  // FORWARD PASS
  const TargetType y_true; // sparse -> shape(y_true) = (1, n_samples)
                           // one hot -> shape(y_true) = (n_classes, n_samples)

  Eigen::MatrixXd y_pred; // shape(y_pred) = (n_classes, n_samples)

  // BACKWARD PASS
  Eigen::MatrixXd d_y_pred;

  double average_loss;

  // Helper for static_assert
  template <typename>
  struct always_false : std::false_type {
  };
};

template <typename TargetType>
Eigen::MatrixXd Loss_CCE<TargetType>::operator()(const Eigen::MatrixXd &predictions)
{
  y_pred = predictions;

  if constexpr (std::is_same_v<TargetType, Eigen::RowVectorXi>) {
    // ASSERTIONS
    const bool valid_y_pred_rows = y_pred.cols() == y_true.cols();
    assert(((valid_y_pred_rows) && "mismatch in column length between y_pred and y_true"));

    const bool valid_y_true_range = y_true.minCoeff() >= 0 &&
                                    y_true.maxCoeff() < y_pred.rows();
    assert(((valid_y_true_range) && "invalid range of values exist in y"));
    // END ASSERTIONS

    const auto y_pred_clip_log = clip_log(y_pred);

    Eigen::MatrixXd losses(1, y_pred.cols());

    for (Eigen::Index i = 0; i < losses.cols(); ++i) {
      losses(0, i) = -y_pred_clip_log(y_true(i), i);
    }

    average_loss = losses.mean();

    return losses;

  } else if constexpr (std::is_same_v<TargetType, Eigen::MatrixXd>) {
    // ASSERTIONS
    const bool valid_y_pred_shape = y_pred.rows() == y_true.rows() &&
                                    y_pred.cols() == y_true.cols();
    assert(((valid_y_pred_shape) && "predictions has invalid shape"));
    // END ASSERTIONS

    const auto y_pred_clip_log = clip_log(y_pred);

    // This code opts for element-wise operations.
    // Doing matrix product in: -(y_true * y_pred_clip_log).diagonal();
    // wastes a lot of computation b/c non-diagonal
    // entries are calculated for no reason
    Eigen::MatrixXd losses = -(y_true.array() * y_pred_clip_log.array()).colwise().sum();

    average_loss = losses.mean();

    return losses;

  } else {
    static_assert(
        always_false<TargetType>::value,
        "TargetType must be either Eigen::RowVectorXi or Eigen::MatrixXd");
  }
}

template <typename TargetType>
void Loss_CCE<TargetType>::backward(const Eigen::MatrixXd &d_next)
{
  // BEGIN ASSERTIONS
  const bool valid_d_next_shape = d_next.rows() == 1 &&
                                  d_next.cols() == y_pred.cols();

  assert(((valid_d_next_shape) && "d_next has invalid shape"));
  // END ASSERTIONS

  if constexpr (std::is_same_v<TargetType, Eigen::RowVectorXi>) {
    Eigen::MatrixXd one_hot(y_pred.rows(), y_pred.cols());
    one_hot.setZero();

    for (Eigen::Index i; i < y_pred.cols(); ++i) {
      one_hot(y_true(i), i) = 1;
    }

    d_y_pred = (-(one_hot.array() / y_pred.array())).rowwise() * d_next.array();
  } else if constexpr (std::is_same_v<TargetType, Eigen::MatrixXd>) {
    d_y_pred = (-(y_true.array() / y_pred.array())).rowwise() * d_next.array();
  } else {
    static_assert(
        always_false<TargetType>::value,
        "TargetType must be either Eigen::RowVectorXi or Eigen::MatrixXd");
  }
};
} // namespace unn

#endif
