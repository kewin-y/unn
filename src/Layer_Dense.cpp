#include <unn/Layer_Dense.hpp>

#include <cassert>

namespace unn
{
Layer_Dense::Layer_Dense(Eigen::Index n_inputs, Eigen::Index n_outputs)
{
  // Intialize with random weights
  weights = Eigen::MatrixXd::Random(n_outputs, n_inputs);
  biases = Eigen::VectorXd::Random(n_outputs);
}

Layer_Dense::Layer_Dense(const Eigen::MatrixXd &weights, const Eigen::VectorXd &biases)
    : weights(weights), biases(biases)
{
}

Eigen::MatrixXd Layer_Dense::operator()(const Eigen::MatrixXd &inputs)
{
  this->inputs = inputs;

  const bool valid_in_rows = inputs.rows() == weights.cols();

  assert(((valid_in_rows) && "inputs has invalid shape"));

  auto outputs = (weights * inputs).colwise() + biases;

  // shape(outputs) = (n_out, n_samples)
  // n_out becomes number of inputs for next layer
  return outputs;
}

void Layer_Dense::backward(const Eigen::MatrixXd &d_next)
{
  // Gradient from next layer should have same shape as this layer's output
  // Because `d_next` is the gradient of `outputs`
  const bool valid_d_next_shape = d_next.rows() == inputs.rows() && d_next.cols() == inputs.cols();

  assert(((valid_d_next_shape) && "d_next has invalid shape"));

  d_inputs = weights.transpose() * d_next;
  d_weights = d_next * inputs.transpose();
  d_biases = d_next.rowwise().sum();
}
} // namespace unn
