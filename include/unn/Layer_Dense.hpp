#ifndef UNN_LAYER_DENSE_HPP
#define UNN_LAYER_DENSE_HPP

#include "unn/Layer.hpp"
#include <Eigen/Core>

namespace unn
{
class Layer_Dense : public Layer
{
private:
  // BACKWARD PASS
  Eigen::MatrixXd d_weights;
  Eigen::MatrixXd d_biases;

public:
  Layer_Dense(Eigen::Index n_in, Eigen::Index n_out);
  Layer_Dense(const Eigen::MatrixXd &weights, const Eigen::VectorXd &biases);

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) override;
  void backward(const Eigen::MatrixXd &d_next) override;

  // FORWARD PASS
  Eigen::MatrixXd weights; // Each row represents
                           // the weights for a specific output neuron.
                           // shape(weights) = (n_out, n_in)

  Eigen::VectorXd biases; // Each entry b_i of `biases` corresponds to the bias of neuron `i`

  // Momentum
  // Intialized to 0x0
  Eigen::MatrixXd weights_velocity;
  Eigen::VectorXd biases_velocity;

  const Eigen::MatrixXd &get_d_weights() const;
  const Eigen::MatrixXd &get_d_biases() const;
  bool velocity_initialized();
};
} // namespace unn

#endif // UNN_LAYER_DENSE_HPP
