#ifndef UNN_LOSS_CCE_HPP
#define UNN_LOSS_CCE_HPP

#include "unn/Loss.hpp"
#include <Eigen/Core>

namespace unn
{
class Loss_CCE : public Loss
{
public:
  Loss_CCE() = default;

  virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &input, const Eigen::RowVectorXi &target) override;
  virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target) override;

  void backward() override;
};
} // namespace unn

#endif
