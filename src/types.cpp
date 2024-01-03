#include "types.hpp"

Axis::Axis(double x_max, double y_max)
    : x{Eigen::ArrayXf::LinSpaced(ic::n_x, 0.5 * ic::dx, x_max - 0.5 * ic::dx)}
    , y{Eigen::ArrayXf::LinSpaced(ic::n_y, 0.5 * ic::dx, y_max - 0.5 * ic::dx)}
{
}

Eigen::MatrixXd add_ghost_cells(Eigen::MatrixXd const&& quantity)
{
  Eigen::MatrixXd result(ic::n_x, ic::n_y + 2);
  result.block(0, 1, quantity.rows(), quantity.cols()) = quantity;
  result.col(0) = result.col(1);
  result.col(ic::n_y + 1) = result.col(ic::n_y);
  return result;
}

PrimitiveQuantities::PrimitiveQuantities() = default;

PrimitiveQuantities::PrimitiveQuantities(Axis const& axis)
{
  auto fill_rho = [](uint8_t i, uint8_t j) {
    return (j < ic::n_y / 2) ? 1. : 2.;
  };

  auto perturbation = [&](uint8_t i, uint8_t j) {
    return 0.0025 * (1. - std::cos(4. * M_PI * axis.x(i))) *
           (1. - std::cos(4. / 3. * M_PI * axis.y(j)));
  };

  auto fill_pressure = [&](uint8_t i, uint8_t j) {
    return ic::p_0 + ic::g * (axis.y(j) - 0.75) * rho(i, j);
  };

  rho =
      add_ghost_cells(Eigen::MatrixXd::NullaryExpr(ic::n_x, ic::n_y, fill_rho));

  v_x = add_ghost_cells(Eigen::MatrixXd::Constant(ic::n_x, ic::n_y, 0.));

  v_y = add_ghost_cells(
      Eigen::MatrixXd::NullaryExpr(ic::n_x, ic::n_y, perturbation));

  pressure = add_ghost_cells(
      Eigen::MatrixXd::NullaryExpr(ic::n_x, ic::n_y, fill_pressure));
}