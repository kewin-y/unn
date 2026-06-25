#include "unn/SGD.hpp"

namespace unn
{
SGD::SGD(double learning_rate, double decay, double momentum)
    : starting_learning_rate{learning_rate}, decay{decay}, momentum{momentum}, learning_rate{learning_rate},
      iterations{0}
{
}

void SGD::update_params(Layer_Dense &layer)
{
  if (momentum != 0.0) {
    // Lazily initialize layer momentum
    if (!layer.velocity_initialized()) {
      const auto biases_size = layer.biases.size();
      const auto weights_rows = layer.weights.rows();
      const auto weights_cols = layer.weights.cols();

      layer.weights_velocity = Eigen::MatrixXd::Zero(weights_rows, weights_cols);
      layer.biases_velocity = Eigen::VectorXd::Zero(biases_size);
    }

    // clang-format off
    // https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    Eigen::MatrixXd weights_velocity =
        momentum * layer.weights_velocity.array()
        - learning_rate * layer.get_d_weights().array();

    Eigen::VectorXd biases_velocity =
       momentum * layer.biases_velocity.array()
       - learning_rate * layer.get_d_biases().array();
    // clang-format on

    layer.weights.array() += weights_velocity.array();
    layer.biases.array() += biases_velocity.array();

    layer.weights_velocity = std::move(weights_velocity);
    layer.biases_velocity = std::move(biases_velocity);

  } else {
    layer.biases.array() += -learning_rate * layer.get_d_biases().array();
    layer.weights.array() += -learning_rate * layer.get_d_weights().array();
  }
}
void SGD::pre_update() { learning_rate = starting_learning_rate / (1 + decay * iterations); }
void SGD::post_update() { iterations++; }
} // namespace unn
