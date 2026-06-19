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
  const double decay;
  const double momentum;

  double learning_rate;
  uint32_t iterations;

public:
  SGD(double learning_rate = 1.0f, double decay = 0.0f, double momentum = 0.0f);

  void update_params(Layer_Dense &layer);
  void pre_update();
  void post_update();
};
} // namespace unn

#endif
