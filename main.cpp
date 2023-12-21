#include <cmath>
#include <iostream>
#include <memory>

#include <Eigen/Core>

constexpr uint8_t n_x = 4;
constexpr uint8_t n_y = 12;
constexpr double x_lim = 0.5;
constexpr double y_lim = 1.5;
constexpr double dx = x_lim / n_x;  // dy = dx
constexpr double volume = dx * dx;
constexpr double p_0 = 2.5;
constexpr double g = -0.1;
constexpr double gamma = 1.4;

struct Axis
{
  Eigen::ArrayXf x_axis;
  Eigen::ArrayXf y_axis;

  Axis(double x_max, double y_max)
      : x_axis{Eigen::ArrayXf::LinSpaced(n_x, 0, x_max)}
      , y_axis{Eigen::ArrayXf::LinSpaced(n_y, 0, y_max)}
  {
  }
};

namespace ghost_manager {
Eigen::MatrixXd add_ghost_cells(Eigen::MatrixXd&& quantity)
{
  Eigen::MatrixXd result(n_x, n_y + 2);
  result.block(0, 1, quantity.rows(), quantity.cols()) = quantity;
  result.col(0) = result.col(1);
  result.col(n_y + 1) = result.col(n_y);
  return result;
}

void set_ghost_cells(Eigen::MatrixXd& quantity)
{
  quantity.col(0) = quantity.col(1);
  quantity.col(n_y + 1) = quantity.col(n_y);
}
}  // namespace ghost_manager

struct PrimitiveQuantities
{
  std::shared_ptr<Axis> xy_axis;
  Eigen::MatrixXd rho;
  Eigen::MatrixXd v_x;
  Eigen::MatrixXd v_y;
  Eigen::MatrixXd pressure;

  PrimitiveQuantities(std::shared_ptr<Axis> axis) : xy_axis{std::move(axis)}
  {
    auto fill_rho = [](uint8_t i, uint8_t j) {
      return (j < n_y / 2) ? 1. : 2.;
    };
    auto perturbation = [&](uint8_t i, uint8_t j) {
      return 0.0025 * (1. - std::cos(4. * M_PI * xy_axis->x_axis[i])) *
             (1. - std::cos(4. / 3. * M_PI * xy_axis->y_axis[j]));
    };
    auto fill_pressure = [&](uint8_t i, uint8_t j) {
      return p_0 + g * (xy_axis->y_axis[j] - 0.75) * rho(i, j);
    };
    rho = ghost_manager::add_ghost_cells(
        Eigen::MatrixXd::NullaryExpr(n_x, n_y, fill_rho));
    v_x =
        ghost_manager::add_ghost_cells(Eigen::MatrixXd::Constant(n_x, n_y, 0.));
    v_y = ghost_manager::add_ghost_cells(
        Eigen::MatrixXd::NullaryExpr(n_x, n_y, perturbation));
    pressure = ghost_manager::add_ghost_cells(
        Eigen::MatrixXd::NullaryExpr(n_x, n_y, fill_pressure));
  }
};

struct ConservedQuantities
{
  Eigen::MatrixXd mass;
  Eigen::MatrixXd momentum_x;
  Eigen::MatrixXd momentum_y;
  Eigen::MatrixXd energy;
};

class Evolver
{
  std::shared_ptr<PrimitiveQuantities> m_primitive;
  std::shared_ptr<ConservedQuantities> m_conserved;

 public:
  Evolver(std::shared_ptr<PrimitiveQuantities> primitive,
          std::shared_ptr<ConservedQuantities> conserved)
      : m_primitive{std::move(primitive)}, m_conserved{std::move(conserved)}
  {
  }

  void update_conserved()
  {
    m_conserved->mass = m_primitive->rho * volume;
    m_conserved->momentum_x = m_primitive->rho * m_primitive->v_x * volume;
    m_conserved->momentum_y = m_primitive->rho * m_primitive->v_y * volume;
    m_conserved->energy = (m_primitive->pressure / (gamma - 1.) +
                           .5 * m_primitive->rho *
                               (m_primitive->v_x * m_primitive->v_x +
                                m_primitive->v_y * m_primitive->v_y)) *
                          volume;
  }

  void update_primitive()
  {
    m_primitive->rho = m_conserved->mass / volume;
    for (uint8_t i{0}; i != n_x; ++i) {
      for (uint8_t j{0}; j != n_y; ++j) {
        m_primitive->v_x(i, j) =
            m_conserved->momentum_x(i, j) / m_primitive->rho(i, j) / volume;
        m_primitive->v_y(i, j) =
            m_conserved->momentum_y(i, j) / m_primitive->rho(i, j) / volume;
      }
    }
    m_primitive->pressure = m_conserved->energy / volume -
                            .5 * m_primitive->rho *
                                (m_primitive->v_x * m_primitive->v_x +
                                 m_primitive->v_y * m_primitive->v_y) *
                                (gamma - 1.);
    ghost_manager::set_ghost_cells(m_primitive->rho);
    ghost_manager::set_ghost_cells(m_primitive->v_x);
    ghost_manager::set_ghost_cells(m_primitive->v_y);
    ghost_manager::set_ghost_cells(m_primitive->pressure);
  }
};

int main()
{
  auto xy_axis = std::make_shared<Axis>(x_lim, y_lim);
  auto primitive_quantities = std::make_shared<PrimitiveQuantities>(xy_axis);
  auto conserved_quantities = std::make_shared<ConservedQuantities>();
}
