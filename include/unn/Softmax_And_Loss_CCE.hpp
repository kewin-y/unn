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

private:
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
  return loss_cce(softmax(inputs));
}

} // namespace unn

#endif
