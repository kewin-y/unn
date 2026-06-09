#ifndef SOFTMAX_AND_LOSS_CCE_HPP
#define SOFTMAX_AND_LOSS_CCE_HPP

#include "unn/Layer.hpp"
#include "unn/Loss_CCE.hpp"
#include "unn/Softmax.hpp"
#include <Eigen/Core>
#include <type_traits>

namespace unn
{
template <typename TargetType> struct Softmax_And_Loss_CCE : Layer {
  Softmax_And_Loss_CCE(const TargetType &targets) : loss_cce{targets}, softmax(), y_true{targets} {};

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) override;
  void backward(const Eigen::MatrixXd &d_next) override;

  const TargetType y_true; // sparse -> shape(y_true) = (1, n_samples)
                           // one hot -> shape(y_true) = (n_classes, n_samples)

  Eigen::MatrixXd inputs; // shape(inputs) = (n_classes, n_samples)

  // BACKWARD PASS
  Eigen::MatrixXd d_inputs;

  double get_average_loss() { return loss_cce.average_loss; }
  const Eigen::MatrixXd &get_preds_ref() const { return softmax.out; }

private:
  // FORWARD PASS
  Loss_CCE<TargetType> loss_cce;
  Softmax softmax;

  // Helper for static_assert
  template <typename> struct always_false : std::false_type {
  };
};

template <typename TargetType>
Eigen::MatrixXd Softmax_And_Loss_CCE<TargetType>::operator()(const Eigen::MatrixXd &inputs)
{
  this->inputs = inputs;
  return loss_cce(softmax(inputs));
}

template <typename TargetType> void Softmax_And_Loss_CCE<TargetType>::backward(const Eigen::MatrixXd &d_next)
{
  // BEGIN ASSERTIONS
  assert(((d_next.rows() == 1 && d_next.cols() == inputs.cols()) && "d_next has invalid shape"));
  // END ASSERTIONS

  // In theory, this check should work because
  // Eigen::RowVectorXi = Eigen::Matrix<int, 1, -1> and
  // Eigen::MatrixXd = Eigen::Matrix<double, -1, -1>
  // Is it the best solution? Idk ask chatgpt
  if constexpr (std::is_same_v<TargetType, Eigen::RowVectorXi>) {
    Eigen::MatrixXd one_hot(inputs.rows(), inputs.cols());
    one_hot.setZero();

    for (Eigen::Index i = 0; i < inputs.cols(); ++i) {
      one_hot(y_true(i), i) = 1;
    }

    d_inputs = (get_preds_ref().array() - one_hot.array()).rowwise() * d_next.row(0).array();

    // Normalize
    d_inputs /= inputs.cols();
  } else if constexpr (std::is_same_v<TargetType, Eigen::MatrixXd>) {
    d_inputs = (get_preds_ref().array() - y_true.array()).rowwise() * d_next.row(0).array();

    // Normalize
    d_inputs /= inputs.cols();
  } else {
    static_assert(always_false<TargetType>::value,
                  "TargetType must be either Eigen::RowVectorXi or Eigen::MatrixXd");
  }
}

} // namespace unn

#endif
