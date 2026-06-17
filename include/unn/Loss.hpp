#ifndef LOSS_HPP
#define LOSS_HPP

#include <Eigen/Core>

namespace unn
{
class Loss
{
protected:
  double average_loss;
  Eigen::MatrixXd target; // shape(target) = (n_classes, n_samples)
                          //
                          // Member functions are expected to compute
                          // the one-hot MatrixXd if given a RowVectorXi.
  Eigen::MatrixXd input;
  Eigen::MatrixXd d_input;

public:
  virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &input, const Eigen::RowVectorXi &target) = 0;
  virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target) = 0;
  virtual void backward() = 0;

  // FORWARD PASS
  const Eigen::MatrixXd &get_target() { return target; }

  const Eigen::MatrixXd &get_input() { return input; } // shape(input) = (n_classes, n_samples)

  // BACKWARD PASS
  const Eigen::MatrixXd &get_d_input() { return d_input; }

  double get_average_loss() { return average_loss; }
};
} // namespace unn

#endif
