#ifndef OPTIMIZER_SGD_HPP
#define OPTIMIZER_SGD_HPP

#include "unn/Layer_Dense.hpp"

namespace unn
{
struct Optimizer_SGD {
  Optimizer_SGD(double learning_rate = 1.0f);

  double learning_rate;

  void update_params(Layer_Dense &layer);
};
} // namespace unn

#endif
