#include <unn/Layer_Dense.hpp>

#include <cassert>

namespace unn
{
Layer_Dense::Layer_Dense(Eigen::Index n_in, Eigen::Index n_out,
                         Eigen::Index n_samples)
    : n_in{n_in}, n_out{n_out}, n_samples{n_samples}
{
  // Intialize with random weights
  weights = Eigen::MatrixXd::Random(n_out, n_in);
  biases = Eigen::VectorXd::Zero(n_out);
}

Eigen::MatrixXd Layer_Dense::operator()(const Eigen::MatrixXd &inputs)
{
  in = inputs;

  const bool valid_in_rows = inputs.rows() == n_in;
  const bool valid_in_cols = inputs.cols() == n_samples;

  assert(((valid_in_rows && valid_in_cols) && "inputs has invalid shape"));

  auto outputs = (weights * inputs).colwise() + biases;

  // shape(outputs) = (n_out, n_samples)
  // n_out becomes number of inputs for next layer
  return outputs;
}

void Layer_Dense::backward(const Eigen::MatrixXd &d_next)
{
  // Gradient from next layer should have same shape as this layer's output
  // Because `d_next` is the gradient of `outputs`
  const bool valid_d_next_rows = d_next.rows() == n_out;
  const bool valid_d_next_cols = d_next.cols() == n_samples;

  assert(((valid_d_next_rows && valid_d_next_cols) && "d_next has invalid shape"));

  d_in = weights.transpose() * d_next;
  d_weights = d_next * in.transpose();
  d_biases = d_next.rowwise().sum();
}
} // namespace unn
