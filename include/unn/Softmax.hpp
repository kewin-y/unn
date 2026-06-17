#ifndef UNN_SOFTMAX_HPP
#define UNN_SOFTMAX_HPP

#include <Eigen/Core>

namespace unn
{
class Softmax
{
private:
  Eigen::MatrixXd output;
  Eigen::MatrixXd d_input;

public:
  Softmax() = default;

  // Maybe? add output member to all layer classes
  // and make the forward pass return a const ref
  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs);
  void backward(const Eigen::MatrixXd &d_next);
  const Eigen::MatrixXd &get_output() const;
};
} // namespace unn

#endif // UNN_SOFTMAX_HPP
