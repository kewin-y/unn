#include <unn/ReLU.hpp>

namespace unn
{
Eigen::MatrixXd ReLU::operator()(const Eigen::MatrixXd &inputs) const
{
  return inputs.cwiseMax(0);
}
} // namespace unn
