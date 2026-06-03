#include <Eigen/Core>
#include <doctest/doctest.h>
#include <unn/Layer_Dense.hpp>

TEST_CASE("Layer_Dense forward pass")
{
  const Eigen::Index n_inputs = 4;
  const Eigen::Index n_outputs = 3;
  const Eigen::Index n_samples = 5;
  Eigen::VectorXd biases(n_outputs);
  Eigen::MatrixXd inputs(n_inputs, n_samples);
  Eigen::MatrixXd weights(n_outputs, n_inputs);
  Eigen::MatrixXd expected(n_outputs, n_samples);

  biases << 3.0, 4.0, 5.0;

  // clang-format off
  weights << 4.5, 4.5, 4.6, 6.3,
             1.7, 8.7, 8.9, 6.5,
             6.7, 7.1, 0.1, 0.6;

  inputs << 6.2, 1.0, 0.3, 2.9, 6.7,
            8.5, 4.1, 3.3, 1.9, 4.2,
            1.0, 8.9, 3.9, 5.6, 3.6,
            0.7, 0.5, 6.0, 1.5, 8.4;

  expected << 78.16,  70.04,  74.94,  59.81, 121.53,
              101.94, 123.83, 106.93, 85.05, 138.57,
              107.41, 42,     34.43,  39.38, 85.11;
  // clang-format on

  unn::Layer_Dense layer(n_inputs, n_outputs, n_samples, weights, biases);

  auto result = layer(inputs);

  CHECK((result.isApprox(expected, 1e-4)));
}
