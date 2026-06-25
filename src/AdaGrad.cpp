#include "unn/AdaGrad.hpp"

namespace unn
{
AdaGrad::AdaGrad(double learning_rate, double epsilon) : learning_rate{learning_rate}, epsilon{epsilon} {}

void AdaGrad::update_params(Layer_Dense &layer)
{
  if (!layer.cache_initialized()) {
    const auto biases_size = layer.biases.size();
    const auto weights_rows = layer.weights.rows();
    const auto weights_cols = layer.weights.cols();

    layer.weights_cache = Eigen::MatrixXd::Zero(weights_rows, weights_cols);
    layer.biases_cache = Eigen::VectorXd::Zero(biases_size);
  }

  layer.weights_cache.array() += (layer.get_d_weights().array() * layer.get_d_weights().array());
  layer.biases_cache.array() += (layer.get_d_biases().array() * layer.get_d_biases().array());

  Eigen::MatrixXd wcof = -learning_rate * (layer.weights_cache.array() + epsilon).cwiseSqrt().cwiseInverse();

  Eigen::MatrixXd bcof = -learning_rate * (layer.biases_cache.array() + epsilon).cwiseSqrt().cwiseInverse();

  layer.weights.array() += wcof.array() * layer.get_d_weights().array();
  layer.biases.array() += wcof.array() * layer.get_d_biases().array();
}

void AdaGrad::post_update() { iterations++; }
} // namespace unn
