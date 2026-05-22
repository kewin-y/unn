#ifndef UNN_LAYER_DENSE_HPP
#define UNN_LAYER_DENSE_HPP

#include "unn/Layer.hpp"
#include <Eigen/Core>

namespace unn
{
struct Layer_Dense : Layer {

  Layer_Dense(Eigen::Index n_in, Eigen::Index n_out, Eigen::Index n_samples);

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) override;
  void backward(const Eigen::MatrixXd &d_next) override;

private:
  const Eigen::Index n_in;      // Number of inputs (features)
  const Eigen::Index n_out;     // Number of output neurons
  const Eigen::Index n_samples; // Batch size

  // FORWARD PASS
  // Each row of `weights` represents the weights for a specific output neuron
  // shape(weights) = (n_out, n_in)
  Eigen::MatrixXd weights;

  Eigen::VectorXd biases; // Each entry b_i of `biases` corresponds to the bias of neuron `i`

  // shape(inputs) = (n_in, n_samples).
  Eigen::MatrixXd in;

  // BACKWARD PASS
  Eigen::MatrixXd d_in;
  Eigen::MatrixXd d_weights;
  Eigen::MatrixXd d_biases;

  // NOTE (hopefully im right im probably wrong)
  // will need grad_in
  // will need grad_weights
  // will need grad_biases
  // will need grad_next (gradient that comes in from the next layer; flows in during backpropagation)
  // Assume grad_next is a matrix whose column vectors are gradient vectors corresponding to a sample
  // i.e., shape(grad_next) = (n_out, n_samples)
  //
  // grad_in = weights^T * grad_next
  // grad_weights = grad_next * inputs^T
  // grad_biases = rowise sum of grad_next
};
} // namespace unn

#endif // UNN_LAYER_DENSE_HPP
