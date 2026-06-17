#include "unn/Loss_CCE.hpp"
#include <Eigen/Core>
#include <doctest/doctest.h>

TEST_CASE("Categorical Cross Entropy loss forward pass (sparse)")
{
  const Eigen::Index n_classes = 3;
  const Eigen::Index n_samples = 3;

  Eigen::MatrixXd predictions(n_classes, n_samples);
  Eigen::RowVectorXi targets(n_samples);
  Eigen::MatrixXd expected(1, n_samples);

  // clang-format off
  targets << 0, 1, 1;
  predictions << 0.7, 0.1, 0.02,
                 0.1, 0.5, 0.9,
                 0.2, 0.4, 0.08;
  expected << 0.356674944, 0.693147181, 0.105360516;
  // clang-format on

  unn::Loss_CCE loss_cce{};
  const auto loss = loss_cce(predictions, targets);

  CHECK(loss.isApprox(expected, 1e-4));
}

TEST_CASE("Categorical Cross Entropy loss forward pass (sparse)")
{
  const Eigen::Index n_classes = 3;
  const Eigen::Index n_samples = 3;

  Eigen::MatrixXd predictions(n_classes, n_samples);
  Eigen::MatrixXd targets(n_classes, n_samples);
  Eigen::MatrixXd expected(1, n_samples);

  // clang-format off
  targets << 1, 0, 0,
             0, 1, 1,
             0, 0, 0;

  predictions << 0.7, 0.1, 0.02,
                 0.1, 0.5, 0.9,
                 0.2, 0.4, 0.08;

  expected << 0.356674944, 0.693147181, 0.105360516;
  // clang-format on

  unn::Loss_CCE loss_cce{};
  const auto loss = loss_cce(predictions, targets);

  CHECK(loss.isApprox(expected, 1e-4));
}
