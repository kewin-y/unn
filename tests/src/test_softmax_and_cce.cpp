#include "unn/Softmax_And_Loss_CCE.hpp"
#include <Eigen/Core>
#include <doctest/doctest.h>

// This test case comes from the book
TEST_CASE("Softmax_And_Loss_CCE backward pass from nnfs")
{
  const Eigen::Index n_classes = 3;
  const Eigen::Index n_samples = 3;

  Eigen::RowVectorXi targets(n_classes);
  Eigen::MatrixXd d_next(1, n_samples);
  Eigen::MatrixXd expected(n_classes, n_samples);

  // clang-format off
  targets << 0, 1, 1;
  d_next << 1, 1, 1;

  unn::Softmax_And_Loss_CCE softmax_cce(targets);

  softmax_cce.inputs.resize(n_classes, n_samples);
  softmax_cce.inputs << 0.7, 0.1, 0.02,
                        0.1, 0.5, 0.9,
                        0.2, 0.4, 0.08;

  expected << -0.1,         0.03333333,  0.06666667,
               0.03333333, -0.16666667,  0.13333333,
               0.00666667, -0.03333333,  0.02666667;
  // clang-format on

  expected.transposeInPlace();

  softmax_cce.backward(d_next);

  CHECK(softmax_cce.d_inputs.isApprox(expected, 1e-4));
}

TEST_CASE("Softmax_And_Loss_CCE forward and backward pass")
{
  const Eigen::Index n_classes = 3;
  const Eigen::Index n_samples = 3;

  Eigen::RowVectorXi targets(n_classes);
  Eigen::MatrixXd d_next(1, n_samples);
  Eigen::MatrixXd logits(n_classes, n_samples);
  Eigen::MatrixXd expected_d_inputs(n_classes, n_samples);

  // clang-format off
  targets << 1, 0, 1;
  d_next << 1, 1, 1;

  unn::Softmax_And_Loss_CCE softmax_cce(targets);

  logits << 2.3, 1.2, 3.3,
            1.1, 3.2, 2.8,
            4.6, 5.1, 9.1;

  expected_d_inputs << 0.76666666, 0.06666666, 1.1,
                       0.03333333, 1.06666666, 0.6,
                       1.53333333, 1.7,        3.03333333;
  // clang-format on

  softmax_cce(logits);
  softmax_cce.backward(d_next);
  CHECK(softmax_cce.d_inputs.isApprox(expected_d_inputs, 1e-4));
}
