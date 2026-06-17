#include "unn/Layer_Dense.hpp"
#include "unn/Optimizer_SGD.hpp"
#include "unn/ReLU.hpp"
#include "unn/Softmax_And_Loss_CCE.hpp"
#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <random>

// Modified from: https://cs231n.github.io/neural-networks-case-study/ w/ the help of Gemini
// See: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
struct Dataset {
  Eigen::MatrixXd X;
  Eigen::VectorXi y;
};

Dataset create_data(int samples, int classes)
{
  int total_samples = samples * classes;

  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(total_samples, 2);
  Eigen::VectorXi y = Eigen::VectorXi::Zero(total_samples);

  std::random_device rd;
  std::mt19937 g(rd());
  std::normal_distribution<double> norm_dist(0.0, 0.2);

  for (int class_number = 0; class_number < classes; ++class_number) {
    int start_ix = samples * class_number;

    for (int i = 0; i < samples; ++i) {
      int current_ix = start_ix + i;

      double r = static_cast<double>(i) / (samples - 1);

      double t_base = (class_number * 4.0) + ((4.0 * i) / (samples - 1));
      double t = t_base + norm_dist(g);

      X(current_ix, 0) = r * std::sin(t * 2.5);
      X(current_ix, 1) = r * std::cos(t * 2.5);

      y(current_ix) = class_number;
    }
  }

  return {X, y};
}

int main()
{
  int samples = 1000;
  int classes = 3;

  Dataset data = create_data(samples, classes);

  auto X = data.X.transpose();
  std::cout << "shape(X) = (" << X.rows() << ", " << X.cols() << ")\n";
  Eigen::RowVectorXi y = data.y.transpose();

  unn::Layer_Dense dense1{2, 64};
  unn::ReLU activation1{};
  unn::Layer_Dense dense2{64, 3};
  unn::Softmax_And_Loss_CCE loss_activation{};
  unn::Optimizer_SGD SGD{};

  // Forward Pass
  auto dense1_out = dense1(X);
  auto activation1_out = activation1(dense1_out);
  auto dense2_out = dense2(activation1_out);

  (void)loss_activation(dense2_out, y);

  std::cout << "Loss: " << loss_activation.get_average_loss() << "\n";

  const auto out_cols = loss_activation.get_softmax_out().cols();
  Eigen::Index maxIndex;
  Eigen::RowVectorXi preds(out_cols);

  for (int j = 0; j < out_cols; j++) {
    loss_activation.get_softmax_out().col(j).maxCoeff(&maxIndex);
    preds(j) = maxIndex;
  }

  int total_correct = (preds.array() == y.array()).count();
  auto accuracy = static_cast<double>(total_correct) / static_cast<double>(out_cols);

  std::cout << "Accuracy: " << accuracy << "\n";

  // Backward Pass

  return 0;
}
