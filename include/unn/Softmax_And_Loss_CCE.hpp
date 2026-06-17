#ifndef SOFTMAX_AND_LOSS_CCE_HPP
#define SOFTMAX_AND_LOSS_CCE_HPP

#include "unn/Loss_CCE.hpp"
#include "unn/Softmax.hpp"
#include <Eigen/Core>

namespace unn
{
class Softmax_And_Loss_CCE
{
private:
  // FORWARD PASS
  Loss_CCE loss_cce;
  Softmax softmax;

  Eigen::MatrixXd d_input;

public:
  Softmax_And_Loss_CCE() = default;

  template <typename T> Eigen::MatrixXd operator()(const Eigen::MatrixXd &input, const T &target)
  {
    static_assert(std::is_same_v<T, Eigen::RowVectorXi> || std::is_same_v<T, Eigen::MatrixXd>,
                  "target must be either Eigen::RowVectorXi or Eigen::MatrixXd");
    return this->loss_cce(this->softmax(input), target);
  }

  void backward()
  {
    const auto &softmax_out = get_softmax_out();
    const auto &target = loss_cce.get_target();
    const auto samples = target.cols();

    d_input = (softmax_out.array() - target.array()) / static_cast<double>(samples);
  }

  double get_average_loss() { return loss_cce.get_average_loss(); }
  const Eigen::MatrixXd &get_d_input() { return d_input; }
  const Eigen::MatrixXd &get_softmax_out() const { return softmax.out; }
};
} // namespace unn

#endif
