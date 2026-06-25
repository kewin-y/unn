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
  this->input = inputs;

  return (weights * inputs).colwise() + biases;
}

// Note shape(d_next) = shape(output)
void Layer_Dense::backward(const Eigen::MatrixXd &d_next)
{
  // This sums d_next * w across each output derivative
  this->d_input = weights.transpose() * d_next;

  // This sums d_next * x across each input
  this->d_weights = d_next * input.transpose();

  this->d_biases = d_next.rowwise().sum();
}

const Eigen::MatrixXd &Layer_Dense::get_d_weights() const { return this->d_weights; }
const Eigen::MatrixXd &Layer_Dense::get_d_biases() const { return this->d_biases; }

bool Layer_Dense::velocity_initialized() const
{
  // You could do *.size() != 0
  return biases_velocity.size() == biases.size() && weights_velocity.rows() == weights.rows() &&
         weights_velocity.cols() == weights.cols();
}

bool Layer_Dense::cache_initialized() const
{
  return biases_cache.size() == biases.size() && weights_velocity.rows() == weights.rows() &&
         weights_cache.cols() == weights.cols();
}
} // namespace unn
