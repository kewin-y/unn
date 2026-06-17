#include "unn/Optimizer_SGD.hpp"

namespace unn
{
Optimizer_SGD::Optimizer_SGD(double learning_rate) : learning_rate{learning_rate} {}

void Optimizer_SGD::update_params(Layer_Dense &layer)
{
  layer.biases.array() -= learning_rate * layer.get_d_biases().array();
  layer.weights.array() -= learning_rate * layer.get_d_weights().array();
}
} // namespace unn
