#include <unn/Layer_Dense.hpp>

#include <cassert>

namespace unn
{
Layer_Dense::Layer_Dense(Eigen::Index n_in, Eigen::Index n_out,
                         Eigen::Index batch_size)
    : n_in{n_in}, n_out{n_out}, batch_size{batch_size}
{
  // Intialize with random weights
  weights = Eigen::MatrixXd::Random(n_out, n_in);
  biases = Eigen::VectorXd::Zero(n_out);
}

Eigen::MatrixXd Layer_Dense::operator()(const Eigen::MatrixXd &inputs) const
{
  const bool valid_in_nrows = inputs.rows() == n_in;
  const bool valid_in_ncols = inputs.cols() == batch_size;

  assert(((valid_in_nrows && valid_in_ncols) && "inputs has invalid size"));

  auto outputs = (weights * inputs).colwise() + biases;

  const bool valid_out_nrows = outputs.rows() == n_out;
  const bool valid_out_ncols = outputs.cols() == batch_size;

  assert(((valid_out_nrows && valid_out_ncols) && "outputs has invalid size"));

  return outputs;
}
} // namespace unn
