#include <cmath>
#include <iostream>

#include <Eigen/Core>

constexpr uint8_t n_x = 64;
constexpr uint8_t n_y = 192;
constexpr double p_0 = 2.5;
constexpr double g = -0.1;
constexpr double gamma = 1.4;

struct Axis
{
  Eigen::ArrayXf x_axis;
  Eigen::ArrayXf y_axis;

  Axis(float x_max, float y_max)
      : x_axis{Eigen::ArrayXf::LinSpaced(n_x, 0, x_max)}
      , y_axis{Eigen::ArrayXf::LinSpaced(n_y, 0, y_max)}
  {
  }
};

Eigen::MatrixXd add_ghost_cells(Eigen::MatrixXd&& quantity)
{
  Eigen::MatrixXd result(n_x, n_y + 2);
  result.block(0, 1, quantity.rows(), quantity.cols()) = quantity;
  result.col(0) = result.col(1);
  result.col(n_y + 1) = result.col(n_y);
  return result;
}

int main()
{
  auto xy_axis = Axis{.5, 1.5};

  auto fill_rho = [](uint8_t i, uint8_t j) -> double {
    return (j < n_y / 2) ? 1. : 2.;
  };
  auto rho = add_ghost_cells(Eigen::MatrixXd::NullaryExpr(n_x, n_y, fill_rho));

  auto perturbation = [&](uint8_t i, uint8_t j) {
    return 0.0025 * (1. - std::cos(4. * M_PI * xy_axis.x_axis[i])) *
           (1. - std::cos(4. / 3. * M_PI * xy_axis.y_axis[j]));
  };

  auto v_x = add_ghost_cells(Eigen::MatrixXd::Constant(n_x, n_y, 0.));

  auto v_y =
      add_ghost_cells(Eigen::MatrixXd::NullaryExpr(n_x, n_y, perturbation));

  auto fill_pressure = [&](uint8_t i, uint8_t j) {
    return p_0 + g * (xy_axis.y_axis[j] - 0.75) * rho(i, j);
  };

  auto pressure =
      add_ghost_cells(Eigen::MatrixXd::NullaryExpr(n_x, n_y, fill_pressure));

  // std::cout << rho << '\n' << v_x << '\n' << v_y << '\n' << pressure << '\n';
}

