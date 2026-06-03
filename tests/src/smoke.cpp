#include <Eigen/Core>
#include <doctest/doctest.h>
#include <unn/Layer_Dense.hpp>
#include <unn/ReLU.hpp>

TEST_CASE("smoke")
{
  unn::ReLU relu{};
  const Eigen::MatrixXd input = Eigen::MatrixXd::Zero(1, 1);
  const auto output = relu(input);
  CHECK(output.size() == 1);
}

