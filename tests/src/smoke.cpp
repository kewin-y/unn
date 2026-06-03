#include <Eigen/Core>
#include <unn/ReLU.hpp>
#include <doctest/doctest.h>

TEST_CASE("smoke") {
  unn::ReLU relu{};
  const Eigen::MatrixXd input = Eigen::MatrixXd::Zero(1, 1);
  const auto output = relu(input);
  CHECK(output.size() == 1);
}
