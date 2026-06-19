#ifndef SGD_HPP
#define SGD_HPP

#include "unn/Layer_Dense.hpp"
#include <cstdint>

namespace unn
{
class SGD
{
private:
  const double starting_learning_rate;
  double learning_rate;
  double decay;
  uint32_t iterations;

public:
  SGD(double learning_rate = 1.0f, double decay = 0.1f);

  void update_params(Layer_Dense &layer);
};
} // namespace unn

#endif
