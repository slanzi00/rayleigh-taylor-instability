#include <cmath>
#include <iostream>
#include <memory>

#include <Eigen/Core>

constexpr uint8_t n_x = 64;
constexpr uint8_t n_y = 192;
constexpr double x_lim = 0.5;
constexpr double y_lim = 1.5;
constexpr double dx = x_lim / n_x;  // dy = dx
constexpr double volume = dx * dx;
constexpr double p_0 = 2.5;
constexpr double g = -0.1;
constexpr double gamma = 1.4;

using MatrixPair = std::pair<Eigen::MatrixXd, Eigen::MatrixXd>;

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

void set_ghost_gradients(MatrixPair& grad)
{
  grad.second.col(0) = -grad.second.col(1);
  grad.second.col(n_y + 1) = -grad.second.col(n_y);
}
}  // namespace ghost_manager

MatrixPair get_gradient(Eigen::MatrixXd const& f)
{
  Eigen::MatrixXd first_f_dx(n_x, n_y + 2);
  first_f_dx << f.bottomRows(n_x - 1), f.topRows(1);

  Eigen::MatrixXd second_f_dx(n_x, n_y + 2);
  second_f_dx << f.bottomRows(1), f.topRows(n_x - 1);

  Eigen::MatrixXd first_f_dy(n_x, n_y + 2);
  first_f_dy << f.rightCols(n_y + 1), f.leftCols(1);

  Eigen::MatrixXd second_f_dy(n_x, n_y + 2);
  second_f_dy << f.rightCols(1), f.leftCols(n_y + 1);

  MatrixPair grad =
      std::make_pair(first_f_dx - second_f_dx, first_f_dy - second_f_dy);
  ghost_manager::set_ghost_gradients(grad);

  return grad;
}

struct SpatialExtrapolated
{
  Eigen::MatrixXd xl;
  Eigen::MatrixXd xr;
  Eigen::MatrixXd yl;
  Eigen::MatrixXd yr;
};

SpatialExtrapolated extrapolate_in_space_to_face(Eigen::MatrixXd const& f,
                                                 MatrixPair const& grad_f)
{
  SpatialExtrapolated spatial_extrapolated{};
  Eigen::MatrixXd f_xl(n_x, n_y + 2);
  f_xl << (f - grad_f.first * dx / 2).bottomRows(n_x - 1),
      (f - grad_f.first * dx / 2).topRows(1);
  spatial_extrapolated.xl = f_xl;

  spatial_extrapolated.xr = f + grad_f.first * dx / 2;

  Eigen::MatrixXd f_yl(n_x, n_y + 2);
  f_yl << (f - grad_f.second * dx / 2).rightCols(n_y + 1),
      (f - grad_f.second * dx / 2).leftCols(1);
  spatial_extrapolated.yl = f_yl;

  spatial_extrapolated.yr = f + grad_f.second * dx / 2;

  return spatial_extrapolated;
}

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
  Eigen::MatrixXd flux_mass_x;
  Eigen::MatrixXd flux_mass_y;
  Eigen::MatrixXd momentum_x;
  Eigen::MatrixXd flux_momentum_x_x;
  Eigen::MatrixXd flux_momentum_x_y;
  Eigen::MatrixXd momentum_y;
  Eigen::MatrixXd flux_momentum_y_x;
  Eigen::MatrixXd flux_momentum_y_y;
  Eigen::MatrixXd energy;
  Eigen::MatrixXd flux_energy_x;
  Eigen::MatrixXd flux_energy_y;
};

class Evolver
{
  std::shared_ptr<PrimitiveQuantities> m_primitive;
  std::shared_ptr<ConservedQuantities> m_conserved;
  double m_dt()
  {
    Eigen::MatrixXd matrix_dt(n_x, n_y + 2);
    for (uint8_t i{0}; i != n_x; ++i) {
      for (uint8_t j{0}; j != n_y + 2; ++j) {
        matrix_dt(i, j) =
            dx / (std::sqrt(gamma * m_primitive->pressure(i, j) /
                            m_primitive->rho(i, j)) +
                  std::sqrt(m_primitive->v_x(i, j) * m_primitive->v_x(i, j) +
                            m_primitive->v_y(i, j) * m_primitive->v_y(i, j)));
      }
    }
    return matrix_dt.minCoeff();
  }

 public:
  Evolver(std::shared_ptr<PrimitiveQuantities> primitive,
          std::shared_ptr<ConservedQuantities> conserved)
      : m_primitive{std::move(primitive)}, m_conserved{std::move(conserved)}
  {
  }

  void update_conserved()
  {
    m_conserved->mass = m_primitive->rho * volume;

    m_conserved->momentum_x =
        (m_primitive->rho.array() * m_primitive->v_x.array()).matrix() * volume;
    m_conserved->momentum_y =
        (m_primitive->rho.array() * m_primitive->v_y.array()).matrix() * volume;

    m_conserved->energy =
        (m_primitive->pressure / (gamma - 1.) +
         .5 * (m_primitive->rho.array() *
               ((m_primitive->v_x.array() * m_primitive->v_x.array()).matrix() +
                (m_primitive->v_y.array() * m_primitive->v_y.array()).matrix())
                   .array())
                  .matrix()) *
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
        m_primitive->pressure(i, j) =
            m_conserved->energy(i, j) / volume -
            .5 * m_primitive->rho(i, j) *
                (m_primitive->v_x(i, j) * m_primitive->v_x(i, j) +
                 m_primitive->v_y(i, j) * m_primitive->v_y(i, j)) *
                (gamma - 1.);
      }
    }

    ghost_manager::set_ghost_cells(m_primitive->rho);
    ghost_manager::set_ghost_cells(m_primitive->v_x);
    ghost_manager::set_ghost_cells(m_primitive->v_y);
    ghost_manager::set_ghost_cells(m_primitive->pressure);
  }

  void add_source_term()
  {
    auto dt = m_dt();
    m_conserved->energy += dt / 2 * m_conserved->momentum_y * g;
    m_conserved->momentum_y += dt / 2 * m_conserved->mass * g;
  }

  void fill_fluxes(SpatialExtrapolated const& rho,
                   SpatialExtrapolated const& v_x,
                   SpatialExtrapolated const& v_y,
                   SpatialExtrapolated const& pressure)
  {
    // left and right energies for its components
    Eigen::MatrixXd energy_xl =
        pressure.xl / (gamma - 1) +
        0.5 * rho.xl.cwiseProduct(v_x.xl.cwiseProduct(v_x.xl) +
                                  v_y.xl.cwiseProduct(v_y.xl));
    Eigen::MatrixXd energy_xr =
        pressure.xr / (gamma - 1) +
        0.5 * rho.xr.cwiseProduct(v_x.xr.cwiseProduct(v_x.xr) +
                                  v_y.xr.cwiseProduct(v_y.xr));
    Eigen::MatrixXd energy_yl =
        pressure.yl / (gamma - 1) +
        0.5 * rho.yl.cwiseProduct(v_x.yl.cwiseProduct(v_x.yl) +
                                  v_y.yl.cwiseProduct(v_y.yl));
    Eigen::MatrixXd energy_yr =
        pressure.yr / (gamma - 1) +
        0.5 * rho.yr.cwiseProduct(v_x.yr.cwiseProduct(v_x.yr) +
                                  v_y.yr.cwiseProduct(v_y.yr));

    // compute star (averaged) states
    Eigen::MatrixXd rho_star_x = 0.5 * (rho.xl + rho.xr);
    Eigen::MatrixXd rho_star_y = 0.5 * (rho.yl + rho.yr);
    Eigen::MatrixXd momentum_x_star_x =
        0.5 * (rho.xl.cwiseProduct(v_x.xl) + rho.xr.cwiseProduct(v_x.xr));
    Eigen::MatrixXd momentum_x_star_y =
        0.5 * (rho.yl.cwiseProduct(v_x.yl) + rho.yl.cwiseProduct(v_x.yr));
    Eigen::MatrixXd momentum_y_star_x =
        0.5 * (rho.xl.cwiseProduct(v_y.xl) + rho.xr.cwiseProduct(v_y.xr));
    Eigen::MatrixXd momentum_y_star_y =
        0.5 * (rho.yl.cwiseProduct(v_y.yl) + rho.yr.cwiseProduct(v_y.yr));
    Eigen::MatrixXd energy_star_x = 0.5 * (energy_xl + energy_xr);
    Eigen::MatrixXd energy_star_y = 0.5 * (energy_yl + energy_yr);
    Eigen::MatrixXd pressure_star_x(n_x, n_y + 2);
    Eigen::MatrixXd pressure_star_y(n_x, n_y + 2);
    for (uint8_t i = 0; i != n_x; ++i) {
      for (uint8_t j = 0; j != n_y + 2; ++j) {
        pressure_star_x(i, j) =
            (gamma - 1) *
            (energy_star_x(i, j) -
             0.5 *
                 (momentum_x_star_x(i, j) * momentum_x_star_x(i, j) +
                  momentum_y_star_x(i, j) * momentum_y_star_x(i, j)) /
                 rho_star_x(i, j));
        pressure_star_y(i, j) =
            (gamma - 1) *
            (energy_star_y(i, j) -
             0.5 *
                 (momentum_x_star_y(i, j) * momentum_x_star_y(i, j) +
                  momentum_y_star_y(i, j) * momentum_y_star_y(i, j)) /
                 rho_star_y(i, j));
      }
    }

    // compute fluxes (local Lax-Friedrichs/Rusanov)
    m_conserved->flux_mass_x = momentum_x_star_x;
    m_conserved->flux_mass_y = momentum_x_star_y;

    m_conserved->flux_momentum_x_x =
        momentum_x_star_y.cwiseProduct(
            momentum_x_star_y.cwiseQuotient(rho_star_x)) +
        pressure_star_x;
    m_conserved->flux_momentum_x_y =
        momentum_x_star_y.cwiseProduct(
            momentum_x_star_y.cwiseQuotient(rho_star_y)) +
        pressure_star_y;

    m_conserved->flux_momentum_y_x = momentum_x_star_x.cwiseProduct(
        momentum_y_star_x.cwiseQuotient(rho_star_x));
    m_conserved->flux_momentum_y_y = momentum_x_star_y.cwiseProduct(
        momentum_y_star_y.cwiseQuotient(rho_star_y));

    m_conserved->flux_energy_x =
        (energy_star_x + pressure_star_x)
            .cwiseProduct(momentum_x_star_x.cwiseQuotient(rho_star_x));
    m_conserved->flux_energy_y =
        (energy_star_y + pressure_star_y)
            .cwiseProduct(momentum_x_star_y.cwiseQuotient(rho_star_y));

    // find wavespeeds
    Eigen::MatrixXd c_xl(n_x, n_y + 2);
    Eigen::MatrixXd c_yl(n_x, n_y + 2);
    Eigen::MatrixXd c_xr(n_x, n_y + 2);
    Eigen::MatrixXd c_yr(n_x, n_y + 2);
    for (uint8_t i = 0; i != n_x; ++i) {
      for (uint8_t j = 0; j != n_y + 2; ++j) {
        c_xl(i, j) = std::sqrt(gamma * pressure.xl(i, j) / rho.xl(i, j)) +
                     std::abs(v_x.xl(i, j));
        c_yl(i, j) = std::sqrt(gamma * pressure.yl(i, j) / rho.yl(i, j)) +
                     std::abs(v_x.yl(i, j));
        c_xr(i, j) = std::sqrt(gamma * pressure.xr(i, j) / rho.xr(i, j)) +
                     std::abs(v_x.xr(i, j));
        c_yr(i, j) = std::sqrt(gamma * pressure.yr(i, j) / rho.yr(i, j)) +
                     std::abs(v_x.yr(i, j));
      }
    }
    Eigen::MatrixXd c_x = c_xl.cwiseMax(c_xr);
    Eigen::MatrixXd c_y = c_yl.cwiseMax(c_yr);

    // add stabilizing diffusive term
    m_conserved->flux_mass_x -= c_x.cwiseProduct(0.5 * (rho.xl - rho.xr));
    m_conserved->flux_mass_y -= c_y.cwiseProduct(0.5 * (rho.yl - rho.yl));
    m_conserved->flux_momentum_x_x -= c_x.cwiseProduct(
        0.5 * (rho.xl.cwiseProduct(v_x.xl) - rho.xr.cwiseProduct(v_x.xr)));
    m_conserved->flux_momentum_x_y -= c_y.cwiseProduct(
        0.5 * (rho.yl.cwiseProduct(v_x.yl) - rho.yr.cwiseProduct(v_x.yr)));
    m_conserved->flux_momentum_y_x -= c_x.cwiseProduct(
        0.5 * (rho.xl.cwiseProduct(v_y.xl) - rho.xr.cwiseProduct(v_y.xr)));
    m_conserved->flux_momentum_y_y -= c_y.cwiseProduct(
        0.5 * (rho.yl.cwiseProduct(v_y.yl) - rho.yr.cwiseProduct(v_y.yr)));
    m_conserved->flux_energy_x -=
        c_x.cwiseProduct(0.5 * (energy_xl - energy_xr));
    m_conserved->flux_energy_y -=
        c_y.cwiseProduct(0.5 * (energy_yl - energy_yr));
  }

  void apply_fluxes()
  {
    auto dt = m_dt();

    // for mass
    m_conserved->mass += -dt * dx * m_conserved->flux_mass_x;

    Eigen::MatrixXd flux_mass_x_shiftl(n_x, n_y + 2);
    flux_mass_x_shiftl << m_conserved->flux_mass_x.bottomRows(1),
        m_conserved->flux_mass_x.topRows(n_x - 1);
    m_conserved->mass += dt * dx * flux_mass_x_shiftl;

    m_conserved->mass += -dt * dx * m_conserved->flux_mass_y;

    Eigen::MatrixXd flux_mass_y_shiftl(n_x, n_y + 2);
    flux_mass_y_shiftl << m_conserved->flux_mass_y.rightCols(1),
        m_conserved->flux_mass_y.leftCols(n_y + 1);
    m_conserved->mass += dt * dx * flux_mass_y_shiftl;

    // for momentum_x
    m_conserved->momentum_x += -dt * dx * m_conserved->flux_momentum_x_x;

    Eigen::MatrixXd flux_momentum_x_x_shiftl(n_x, n_y + 2);
    flux_momentum_x_x_shiftl << m_conserved->flux_momentum_x_x.bottomRows(1),
        m_conserved->flux_momentum_x_x.topRows(n_x - 1);
    m_conserved->momentum_x += dt * dx * flux_momentum_x_x_shiftl;

    m_conserved->momentum_x += -dt * dx * m_conserved->flux_momentum_x_y;

    Eigen::MatrixXd flux_momentum_x_y_shiftl(n_x, n_y + 2);
    flux_momentum_x_y_shiftl << m_conserved->flux_momentum_x_y.rightCols(1),
        m_conserved->flux_momentum_x_y.leftCols(n_y + 1);
    m_conserved->momentum_x += dt * dx * flux_momentum_x_y_shiftl;

    // for momentum_y
    m_conserved->momentum_y += -dt * dx * m_conserved->flux_momentum_y_x;

    Eigen::MatrixXd flux_momentum_y_x_shiftl(n_x, n_y + 2);
    flux_momentum_y_x_shiftl << m_conserved->flux_momentum_y_x.bottomRows(1),
        m_conserved->flux_momentum_y_x.topRows(n_x - 1);
    m_conserved->momentum_y += dt * dx * flux_momentum_y_x_shiftl;

    m_conserved->momentum_y += -dt * dx * m_conserved->flux_momentum_y_y;

    Eigen::MatrixXd flux_momentum_y_y_shiftl(n_x, n_y + 2);
    flux_momentum_y_y_shiftl << m_conserved->flux_momentum_y_y.rightCols(1),
        m_conserved->flux_momentum_y_y.leftCols(n_y + 1);
    m_conserved->momentum_y += dt * dx * flux_momentum_y_y_shiftl;

    // for energy
    m_conserved->energy += -dt * dx * m_conserved->flux_energy_x;

    Eigen::MatrixXd flux_energy_x_shiftl(n_x, n_y + 2);
    flux_energy_x_shiftl << m_conserved->flux_energy_x.bottomRows(1),
        m_conserved->flux_energy_x.topRows(n_x - 1);
    m_conserved->energy += dt * dx * flux_energy_x_shiftl;

    m_conserved->energy += -dt * dx * m_conserved->flux_energy_y;

    Eigen::MatrixXd flux_energy_y_shiftl(n_x, n_y + 2);
    flux_energy_y_shiftl << m_conserved->flux_energy_y.rightCols(1),
        m_conserved->flux_energy_y.leftCols(n_y + 1);
    m_conserved->energy += dt * dx * flux_energy_y_shiftl;
  }

  void run_model_for(uint16_t duration_time)
  {
    double time = 0.;
    update_conserved();
    do {
      update_primitive();
      auto dt = m_dt();
      std::cout << dt << '\n';
      add_source_term();
      update_primitive();
      MatrixPair grad_rho = get_gradient(m_primitive->rho);
      MatrixPair grad_v_x = get_gradient(m_primitive->v_x);
      MatrixPair grad_v_y = get_gradient(m_primitive->v_y);
      MatrixPair grad_pressure = get_gradient(m_primitive->pressure);
      Eigen::MatrixXd rho_prime(n_x, n_y + 2);
      Eigen::MatrixXd v_x_prime(n_x, n_y + 2);
      Eigen::MatrixXd v_y_prime(n_x, n_y + 2);
      Eigen::MatrixXd pressure_prime(n_x, n_y + 2);

      for (uint8_t i{0}; i != n_x; ++i) {
        for (uint8_t j{0}; j != n_y + 2; ++j) {
          rho_prime(i, j) =
              m_primitive->rho(i, j) -
              0.5 * dt *
                  (m_primitive->v_x(i, j) * grad_rho.first(i, j) +
                   m_primitive->v_y(i, j) * grad_rho.second(i, j) +
                   m_primitive->rho(i, j) * grad_v_x.first(i, j) +
                   m_primitive->rho(i, j) * grad_v_y.second(i, j));

          double inv_rho = 1. / m_primitive->rho(i, j);
          v_x_prime(i, j) =
              m_primitive->v_x(i, j) -
              0.5 * dt *
                  (m_primitive->v_x(i, j) * grad_v_x.first(i, j) +
                   m_primitive->v_y(i, j) * grad_v_x.second(i, j) +
                   inv_rho * grad_pressure.first(i, j));

          v_y_prime(i, j) =
              m_primitive->v_y(i, j) -
              0.5 * dt *
                  (m_primitive->v_x(i, j) * grad_v_y.first(i, j) +
                   m_primitive->v_y(i, j) * grad_v_y.second(i, j) +
                   inv_rho * grad_pressure.second(i, j));

          pressure_prime(i, j) =
              m_primitive->pressure(i, j) -
              0.5 * dt *
                  (gamma * m_primitive->pressure(i, j) *
                       (grad_v_x.first(i, j) + grad_v_y.second(i, j)) +
                   m_primitive->v_x(i, j) * grad_pressure.first(i, j) +
                   m_primitive->v_y(i, j) * grad_pressure.second(i, j));
        }
      }
      SpatialExtrapolated se_rho =
          extrapolate_in_space_to_face(rho_prime, grad_rho);
      SpatialExtrapolated se_v_x =
          extrapolate_in_space_to_face(v_x_prime, grad_v_x);
      SpatialExtrapolated se_v_y =
          extrapolate_in_space_to_face(v_y_prime, grad_v_y);
      SpatialExtrapolated se_pressure =
          extrapolate_in_space_to_face(pressure_prime, grad_pressure);

      fill_fluxes(se_rho, se_v_x, se_v_y, se_pressure);
      apply_fluxes();
      add_source_term();
      time += dt;
    } while (time < duration_time);
  }
};

int main()
{
  auto xy_axis = std::make_shared<Axis>(x_lim, y_lim);
  auto primitive_quantities = std::make_shared<PrimitiveQuantities>(xy_axis);
  auto conserved_quantities = std::make_shared<ConservedQuantities>();
  auto system_evolver = Evolver{primitive_quantities, conserved_quantities};
  system_evolver.run_model_for(15);
}
