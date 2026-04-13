#include <Eigen/Core>

#include <unn/ReLU.hpp>

int main()
{
  unn::ReLU relu{};
  const Eigen::MatrixXd input = Eigen::MatrixXd::Zero(1, 1);
  const auto output = relu(input);
  return output.size() == 1 ? 0 : 1;
}
