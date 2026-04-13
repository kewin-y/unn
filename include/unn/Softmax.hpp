#ifndef UNN_SOFTMAX_HPP
#define UNN_SOFTMAX_HPP

#include <Eigen/Core>

namespace unn
{
struct Softmax {
  Softmax() = default;
  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) const;
};
} // namespace unn

#endif // UNN_SOFTMAX_HPP
