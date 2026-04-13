#include <Eigen/Core>
#include <iostream>

#include <unn/Layer_Dense.hpp>
#include <unn/Loss_CCE.hpp>
#include <unn/ReLU.hpp>
#include <unn/Softmax.hpp>

int main()
{
  const Eigen::Index nout = 2;
  const Eigen::Index nin = 4;
  const Eigen::Index samples = 2;

  unn::Layer_Dense layer1{nin, nout, samples};
  unn::ReLU relu{};
  unn::Softmax sm{};
  unn::Loss_CCE cce{};

  const Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(nin, samples);

  const auto dense_output = layer1(inputs);
  const auto relu_output = relu(dense_output);
  const auto softmax_output = sm(relu_output);

  const Eigen::Matrix3d logits{
      {0.7, 0.1, 0.02},
      {0.1, 0.5, 0.9},
      {0.2, 0.4, 0.08},
  };

  const Eigen::VectorXi class_targets = Eigen::Vector3i{0, 1, 1};
  const auto loss = cce(logits, class_targets);

  std::cout << "Softmax output:\n"
            << softmax_output << "\n\n";
  std::cout << "Loss:\n"
            << loss << "\n";
  return 0;
}
