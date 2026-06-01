#ifndef UNN_LAYER_DENSE_HPP
#define UNN_LAYER_DENSE_HPP

#include "unn/Layer.hpp"
#include <Eigen/Core>

namespace unn
{
struct Layer_Dense : Layer {

  Layer_Dense(Eigen::Index n_in, Eigen::Index n_out, Eigen::Index n_samples);

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) override;
  void backward(const Eigen::MatrixXd &d_next) override;

private:
  const Eigen::Index m_n_in;      // Number of inputs (features)
  const Eigen::Index m_n_out;     // Number of output neurons
  const Eigen::Index m_n_samples; // Batch size

  // FORWARD PASS
  Eigen::MatrixXd m_weights; // Each row represents
                             // the weights for a specific output neuron.
                             // shape(weights) = (n_out, n_in)

  Eigen::VectorXd m_biases; // Each entry b_i of `biases` corresponds to the bias of neuron `i`

  Eigen::MatrixXd m_inputs; // shape(inputs) = (n_in, n_samples)

  // BACKWARD PASS
  Eigen::MatrixXd m_d_in;
  Eigen::MatrixXd m_d_weights;
  Eigen::MatrixXd m_d_biases;
};
} // namespace unn

#endif // UNN_LAYER_DENSE_HPP
