#include "unn/Softmax.hpp"
#include <Eigen/Core>
#include <doctest/doctest.h>

TEST_CASE("Softmax forward pass")
{
  const Eigen::Index n_classes = 3;
  const Eigen::Index n_samples = 3;

  Eigen::MatrixXd d_next(1, n_samples);
  Eigen::MatrixXd logits(n_classes, n_samples);
  Eigen::MatrixXd expected(n_classes, n_samples);

  // clang-format off
  d_next << 1, 1, 1;

  unn::Softmax softmax{};

  logits << 2.3, 1.2, 3.3,
            1.1, 3.2, 2.8,
            4.6, 5.1, 9.1;

  expected << 0.088688833, 0.017303581, 0.0030129,
              0.026712563, 0.127857132, 0.001827416,
              0.884598604, 0.854839287, 0.995159683;
  // clang-format off

  const auto softmax_outputs = softmax(logits);
  CHECK(softmax_outputs.isApprox(expected, 1e-4));
}
