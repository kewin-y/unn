#ifndef UNN_LAYER_DENSE_HPP
#define UNN_LAYER_DENSE_HPP

#include "unn/Layer.hpp"
#include <Eigen/Core>

namespace unn
{
struct Layer_Dense : Layer {

  Layer_Dense(Eigen::Index n_in, Eigen::Index n_out);
  Layer_Dense(const Eigen::MatrixXd &weights, const Eigen::VectorXd &biases);

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) override;
  void backward(const Eigen::MatrixXd &d_next) override;

private:
  // FORWARD PASS
  Eigen::MatrixXd weights; // Each row represents
                           // the weights for a specific output neuron.
                           // shape(weights) = (n_out, n_in)

  Eigen::VectorXd biases; // Each entry b_i of `biases` corresponds to the bias of neuron `i`

  Eigen::MatrixXd inputs; // shape(inputs) = (n_in, n_samples)

  // BACKWARD PASS
  Eigen::MatrixXd d_inputs;
  Eigen::MatrixXd d_weights;
  Eigen::MatrixXd d_biases;
};
} // namespace unn

#endif // UNN_LAYER_DENSE_HPP
