#ifndef ADAGRAD_HPP
#define ADAGRAD_HPP

#include "unn/Layer_Dense.hpp"
#include <cstdint>

namespace unn
{
class AdaGrad
{
private:
  const double learning_rate;
  const double epsilon;

  uint32_t iterations;

public:
  AdaGrad(double learning_rate = 1.0f, double epsilon = 1e-8);

  void update_params(Layer_Dense &layer);
  void post_update();
};
} // namespace unn

#endif
