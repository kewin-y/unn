#include <unn/Layer_Dense.hpp>

#include <cassert>

namespace unn
{
Layer_Dense::Layer_Dense(Eigen::Index n_in, Eigen::Index n_out,
                         Eigen::Index n_samples)
    : m_n_in{n_in}, m_n_out{n_out}, m_n_samples{n_samples}
{
  // Intialize with random weights
  m_weights = Eigen::MatrixXd::Random(n_out, n_in);
  m_biases = Eigen::VectorXd::Zero(n_out);
}

Eigen::MatrixXd Layer_Dense::operator()(const Eigen::MatrixXd &inputs)
{
  m_inputs = inputs;

  const bool valid_in_rows = inputs.rows() == m_n_in;
  const bool valid_in_cols = inputs.cols() == m_n_samples;

  assert(((valid_in_rows && valid_in_cols) && "inputs has invalid shape"));

  auto outputs = (m_weights * inputs).colwise() + m_biases;

  // shape(outputs) = (n_out, n_samples)
  // n_out becomes number of inputs for next layer
  return outputs;
}

void Layer_Dense::backward(const Eigen::MatrixXd &d_next)
{
  // Gradient from next layer should have same shape as this layer's output
  // Because `d_next` is the gradient of `outputs`
  const bool valid_d_next_rows = d_next.rows() == m_n_out;
  const bool valid_d_next_cols = d_next.cols() == m_n_samples;

  assert(((valid_d_next_rows && valid_d_next_cols) && "d_next has invalid shape"));

  m_d_in = m_weights.transpose() * d_next;
  m_d_weights = d_next * m_inputs.transpose();
  m_d_biases = d_next.rowwise().sum();
}
} // namespace unn
