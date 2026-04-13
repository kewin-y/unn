#ifndef UNN_LAYER_DENSE_HPP
#define UNN_LAYER_DENSE_HPP

#include <Eigen/Core>

namespace unn
{
struct Layer_Dense {
  const Eigen::Index n_in;
  const Eigen::Index n_out;
  const Eigen::Index batch_size;

  Eigen::MatrixXd weights;
  Eigen::VectorXd biases;

  Layer_Dense(Eigen::Index n_in, Eigen::Index n_out, Eigen::Index batch_size);

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) const;
};
} // namespace unn

#endif // UNN_LAYER_DENSE_HPP
