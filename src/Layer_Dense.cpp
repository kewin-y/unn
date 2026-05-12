#include <unn/Layer_Dense.hpp>

#include <cassert>

namespace unn
{
Layer_Dense::Layer_Dense(Eigen::Index n_in, Eigen::Index n_out,
                         Eigen::Index n_samples)
    : n_in{n_in}, n_out{n_out}, n_samples{n_samples}
{
  // Intialize with random weights
  weights = Eigen::MatrixXd::Random(n_out, n_in);
  biases = Eigen::VectorXd::Zero(n_out);
}

Eigen::MatrixXd Layer_Dense::operator()(const Eigen::MatrixXd &inputs)
{
  const bool valid_in_nrows = inputs.rows() == n_in;
  const bool valid_in_ncols = inputs.cols() == n_samples;

  assert(((valid_in_nrows && valid_in_ncols) && "inputs has invalid size"));

  auto outputs = (weights * inputs).colwise() + biases;

  const bool valid_out_nrows = outputs.rows() == n_out;
  const bool valid_out_ncols = outputs.cols() == n_samples;

  assert(((valid_out_nrows && valid_out_ncols) && "outputs has invalid size"));

  in = inputs;

  // shape(outputs) = (n_out, n_samples)
  // n_out becomes number of inputs for next layer
  return outputs;
}
} // namespace unn
