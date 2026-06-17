#include <unn/Softmax.hpp>

namespace unn
{
Eigen::MatrixXd Softmax::operator()(const Eigen::MatrixXd &inputs)
{
  // Row vector containing the maximum input of each sample
  Eigen::RowVectorXd maxes = inputs.colwise().maxCoeff();

  // Adjust inputs
  Eigen::MatrixXd adjusted = inputs.rowwise() - maxes;

  // Apply exponential
  adjusted = adjusted.array().exp();

  // Get sum of each sample
  Eigen::RowVectorXd s = adjusted.colwise().sum();

  // Softmax (hopefully)
  this->output = adjusted.array().rowwise() / s.array();

  return output;
}
void Softmax::backward(const Eigen::MatrixXd &d_next)
{
  this->d_input.resize(output.rows(), output.cols());

  for (Eigen::Index j = 0; j < output.cols(); j++) {
    Eigen::MatrixXd jacobian =
        static_cast<Eigen::MatrixXd>(output.col(j).asDiagonal()) - (output.col(j) * output.col(j).transpose());
    this->d_input.col(j) = jacobian * d_next.col(j);
  }
}
const Eigen::MatrixXd &Softmax::get_output() const { return output; }
} // namespace unn
