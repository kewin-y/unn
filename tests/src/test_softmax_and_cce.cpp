#include "unn/Softmax_And_Loss_CCE.hpp"
#include <Eigen/Core>
#include <Eigen/src/Core/Matrix.h>
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

  // Hmm
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
