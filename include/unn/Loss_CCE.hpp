#ifndef UNN_LOSS_CCE_HPP
#define UNN_LOSS_CCE_HPP

#include <Eigen/Core>

namespace unn
{
struct Loss_CCE
{
  Loss_CCE() = default;

  Eigen::VectorXd operator()(const Eigen::MatrixXd &y_pred, const Eigen::VectorXi &y) const;

  Eigen::VectorXd operator()(const Eigen::MatrixXd &y_pred, const Eigen::MatrixXd &y) const;
};
} // namespace unn

#endif // UNN_LOSS_CCE_HPP
