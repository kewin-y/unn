#ifndef SOFTMAX_AND_LOSS_CCE_HPP
#define SOFTMAX_AND_LOSS_CCE_HPP

#include "unn/Layer.hpp"
#include "unn/Loss_CCE.hpp"
#include "unn/Softmax.hpp"
#include <Eigen/Core>
#include <type_traits>

namespace unn
{
template <typename TargetType>
struct Softmax_And_Loss_CCE : Layer {
  Softmax_And_Loss_CCE<TargetType>(const TargetType &targets) : y_true{targets}, loss_cce{targets}, softmax(){};

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) override;
  void backward(const Eigen::MatrixXd &d_next) override;

  // FORWARD PASS
  Loss_CCE<TargetType> loss_cce;
  Softmax softmax;

  const TargetType y_true; // sparse -> shape(y_true) = (1, n_samples)
                             // one hot -> shape(y_true) = (n_classes, n_samples)

  Eigen::MatrixXd inputs; // shape(inputs) = (n_classes, n_samples)

  // BACKWARD PASS
  Eigen::MatrixXd d_inputs;

  // Helper for static_assert
  template <typename>
  struct always_false : std::false_type {
  };
};

template <typename TargetType>
Eigen::MatrixXd Softmax_And_Loss_CCE<TargetType>::operator()(const Eigen::MatrixXd &inputs)
{
  this->inputs = inputs;
  return loss_cce(softmax(inputs));
}

template <typename TargetType>
void Softmax_And_Loss_CCE<TargetType>::backward(const Eigen::MatrixXd &d_next)
{
  // BEGIN ASSERTIONS
  const bool valid_d_next_shape = d_next.rows() == 1 &&
                                  d_next.cols() == inputs.cols();

  assert(((valid_d_next_shape) && "d_next has invalid shape"));
  // END ASSERTIONS

  if constexpr (std::is_same_v<TargetType, Eigen::RowVectorXi>) {
    Eigen::MatrixXd one_hot(inputs.rows(), inputs.cols());
    one_hot.setZero();

    for (Eigen::Index i; i < inputs.cols(); ++i) {
      one_hot(y_true(i), i) = 1;
    }

    d_inputs = (inputs.array() - one_hot.array()) * d_next.array();

    // Normalize
    d_inputs /= inputs.cols();
  } else if constexpr (std::is_same_v<TargetType, Eigen::MatrixXd>) {
    d_inputs = (inputs.array() - y_true.array()) * d_next.array();

    // Normalize
    d_inputs /= inputs.cols();
  } else {
    static_assert(
        always_false<TargetType>::value,
        "TargetType must be either Eigen::RowVectorXi or Eigen::MatrixXd");
  }
}

} // namespace unn

#endif
