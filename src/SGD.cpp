#include "unn/SGD.hpp"

namespace unn
{
SGD::SGD(double learning_rate, double decay)
    : starting_learning_rate{learning_rate}, learning_rate{learning_rate}, decay{decay}, iterations{0}
{
}

void SGD::update_params(Layer_Dense &layer)
{
  learning_rate = starting_learning_rate / (1 + decay * iterations);

  layer.biases.array() -= learning_rate * layer.get_d_biases().array();
  layer.weights.array() -= learning_rate * layer.get_d_weights().array();

  iterations += 1;
}
} // namespace unn
