#include <cmath>
#include <iostream>
#include <memory>

#include "types.hpp"

ConservedQuantities get_conserved(PrimitiveQuantities const&);

void set_ghost_cells(PrimitiveQuantities&);

PrimitiveQuantities get_primitive(ConservedQuantities const&);

double get_dt(PrimitiveQuantities const&);

void add_source_term(ConservedQuantities&, double);

void set_ghost_gradients(MatrixPair&);

MatrixPair get_gradient(Eigen::MatrixXd const&);

SpatialExtrapolated extrapolate_in_space_to_face(Eigen::MatrixXd const&,
                                                 MatrixPair const&);

ConservedQuantities get_flux(Eigen::MatrixXd const&,
                             Eigen::MatrixXd const&,
                             Eigen::MatrixXd const&,
                             Eigen::MatrixXd const&,
                             Eigen::MatrixXd const&,
                             Eigen::MatrixXd const&,
                             Eigen::MatrixXd const&,
                             Eigen::MatrixXd const&);

Eigen::MatrixXd apply_fluxes(Eigen::MatrixXd&,
                             Eigen::MatrixXd const&,
                             Eigen::MatrixXd const&,
                             double);

void print_matrix(Eigen::MatrixXd const&);

int main()
{
  auto axis = Axis{ic::x_lim, ic::y_lim};
  auto primitives = PrimitiveQuantities{axis};
  auto conserved = get_conserved(primitives);
  double t = 0.;
  do {
    primitives = get_primitive(conserved);
    auto dt = get_dt(primitives);
    add_source_term(conserved, dt / 2.);
    primitives = get_primitive(conserved);

    // calculate gradients
    auto grad_rho = get_gradient(primitives.rho);
    auto grad_v_x = get_gradient(primitives.v_x);
    auto grad_v_y = get_gradient(primitives.v_y);
    auto grad_pressure = get_gradient(primitives.pressure);

    // extrapolate half step in time
    auto rho_prime =
        primitives.rho - 0.5 * dt *
                             (primitives.v_x.cwiseProduct(grad_rho.first) +
                              primitives.rho.cwiseProduct(grad_v_x.first) +
                              primitives.v_y.cwiseProduct(grad_rho.second) +
                              primitives.rho.cwiseProduct(grad_v_y.second));

    auto v_x_prime = primitives.v_x -
                     0.5 * dt *
                         (primitives.v_x.cwiseProduct(grad_v_x.first) +
                          primitives.v_y.cwiseProduct(grad_v_x.second) +
                          grad_pressure.first.cwiseQuotient(primitives.rho));

    auto v_y_prime = primitives.v_y -
                     0.5 * dt *
                         (primitives.v_x.cwiseProduct(grad_v_y.first) +
                          primitives.v_y.cwiseProduct(grad_v_y.second) +
                          grad_pressure.second.cwiseQuotient(primitives.rho));

    auto pressure_prime =
        primitives.pressure -
        0.5 * dt *
            (ic::gamma * primitives.pressure.cwiseProduct(grad_v_x.first +
                                                          grad_v_y.second) +
             primitives.v_x.cwiseProduct(grad_pressure.first) +
             primitives.v_y.cwiseProduct(grad_pressure.second));

    SpatialExtrapolated se_rho =
        extrapolate_in_space_to_face(rho_prime, grad_rho);

    SpatialExtrapolated se_v_x =
        extrapolate_in_space_to_face(v_x_prime, grad_v_x);

    SpatialExtrapolated se_v_y =
        extrapolate_in_space_to_face(v_y_prime, grad_v_y);

    SpatialExtrapolated se_pressure =
        extrapolate_in_space_to_face(pressure_prime, grad_pressure);

    auto fluxes_x = get_flux(se_rho.xl,
                             se_rho.xr,
                             se_v_x.xl,
                             se_v_x.xr,
                             se_v_y.xl,
                             se_v_y.xr,
                             se_pressure.xl,
                             se_pressure.xr);

    auto fluxes_y = get_flux(se_rho.yl,
                             se_rho.yr,
                             se_v_y.yl,
                             se_v_y.yr,
                             se_v_x.yl,
                             se_v_x.yr,
                             se_pressure.yl,
                             se_pressure.yr);

    conserved.mass =
        apply_fluxes(conserved.mass, fluxes_x.mass, fluxes_y.mass, dt);
    conserved.momentum_x = apply_fluxes(
        conserved.momentum_x, fluxes_x.momentum_x, fluxes_y.momentum_y, dt);
    conserved.momentum_y = apply_fluxes(
        conserved.momentum_y, fluxes_x.momentum_y, fluxes_y.momentum_x, dt);
    conserved.energy =
        apply_fluxes(conserved.energy, fluxes_x.energy, fluxes_y.energy, dt);

    add_source_term(conserved, dt / 2.);
    
    print_matrix(primitives.rho);
    t += dt;
  } while (t < ic::t_end);
}

ConservedQuantities get_conserved(PrimitiveQuantities const& primitive)
{
  auto conserved = ConservedQuantities{};
  conserved.mass = primitive.rho * ic::volume;
  conserved.momentum_x = primitive.rho.cwiseProduct(primitive.v_x) * ic::volume;
  conserved.momentum_y = primitive.rho.cwiseProduct(primitive.v_y) * ic::volume;
  conserved.energy = (primitive.pressure / (ic::gamma - 1.) +
                      .5 * primitive.rho.cwiseProduct(
                               primitive.v_x.cwiseProduct(primitive.v_x) +
                               primitive.v_y.cwiseProduct(primitive.v_y))) *
                     ic::volume;
  return conserved;
}

void set_ghost_cells(PrimitiveQuantities& primitive)
{
  primitive.rho.col(0) = primitive.rho.col(1);
  primitive.v_x.col(0) = primitive.v_x.col(1);
  primitive.v_y.col(0) = -primitive.v_y.col(1);
  primitive.pressure.col(0) = primitive.pressure.col(1);

  primitive.rho.col(ic::n_y + 1) = primitive.rho.col(ic::n_y);
  primitive.v_x.col(ic::n_y + 1) = primitive.v_x.col(ic::n_y);
  primitive.v_y.col(ic::n_y + 1) = -primitive.v_y.col(ic::n_y);
  primitive.pressure.col(ic::n_y + 1) = primitive.pressure.col(ic::n_y);
}

PrimitiveQuantities get_primitive(ConservedQuantities const& conserved)
{
  auto primitive = PrimitiveQuantities{};
  primitive.rho = conserved.mass / ic::volume;
  primitive.v_x =
      (1. / ic::volume) * conserved.momentum_x.cwiseQuotient(primitive.rho);
  primitive.v_y =
      (1. / ic::volume) * conserved.momentum_y.cwiseQuotient(primitive.rho);
  primitive.pressure = (conserved.energy / ic::volume -
                        0.5 * primitive.rho.cwiseProduct(
                                  primitive.v_x.cwiseProduct(primitive.v_x) +
                                  primitive.v_y.cwiseProduct(primitive.v_y))) *
                       (ic::gamma - 1.);
  set_ghost_cells(primitive);
  return primitive;
}

double get_dt(PrimitiveQuantities const& primitive)
{
  auto fill_m_dt = [&](uint32_t i, uint32_t j) {
    return ic::dx / (std::sqrt(ic::gamma * primitive.pressure(i, j) /
                               primitive.rho(i, j)) +
                     std::sqrt(primitive.v_x(i, j) * primitive.v_x(i, j) +
                               primitive.v_y(i, j) * primitive.v_y(i, j)));
  };
  auto m_dt = Eigen::MatrixXd::NullaryExpr(ic::n_x, ic::n_y + 2, fill_m_dt);
  return 0.4 * m_dt.minCoeff();
}

void add_source_term(ConservedQuantities& conserved, double dt)
{
  conserved.energy += dt * conserved.momentum_y * ic::g;
  conserved.momentum_y += dt * conserved.mass * ic::g;
}

void set_ghost_gradients(MatrixPair& grad)
{
  grad.second.col(0) = -grad.second.col(1);
  grad.second.col(ic::n_y + 1) = -grad.second.col(ic::n_y);
}

MatrixPair get_gradient(Eigen::MatrixXd const& f)
{
  Eigen::MatrixXd first_f_dx(ic::n_x, ic::n_y + 2);
  first_f_dx << f.bottomRows(ic::n_x - 1), f.topRows(1);

  Eigen::MatrixXd second_f_dx(ic::n_x, ic::n_y + 2);
  second_f_dx << f.bottomRows(1), f.topRows(ic::n_x - 1);

  Eigen::MatrixXd first_f_dy(ic::n_x, ic::n_y + 2);
  first_f_dy << f.rightCols(ic::n_y + 1), f.leftCols(1);

  Eigen::MatrixXd second_f_dy(ic::n_x, ic::n_y + 2);
  second_f_dy << f.rightCols(1), f.leftCols(ic::n_y + 1);

  MatrixPair grad = std::make_pair((first_f_dx - second_f_dx) / (2. * ic::dx),
                                   (first_f_dy - second_f_dy) / (2. * ic::dx));
  set_ghost_gradients(grad);
  return grad;
}

SpatialExtrapolated extrapolate_in_space_to_face(Eigen::MatrixXd const& f,
                                                 MatrixPair const& grad_f)
{
  auto spatial_extrapolated = SpatialExtrapolated{};

  Eigen::MatrixXd f_xl(ic::n_x, ic::n_y + 2);
  f_xl << (f - grad_f.first * ic::dx / 2.).bottomRows(ic::n_x - 1),
      (f - grad_f.first * ic::dx / 2.).topRows(1);
  spatial_extrapolated.xl = f_xl;

  spatial_extrapolated.xr = f + grad_f.first * ic::dx / 2.;

  Eigen::MatrixXd f_yl(ic::n_x, ic::n_y + 2);
  f_yl << (f - grad_f.second * ic::dx / 2.).rightCols(ic::n_y + 1),
      (f - grad_f.second * ic::dx / 2.).leftCols(1);
  spatial_extrapolated.yl = f_yl;

  spatial_extrapolated.yr = f + grad_f.second * ic::dx / 2.;

  return spatial_extrapolated;
}

ConservedQuantities get_flux(Eigen::MatrixXd const& rho_l,
                             Eigen::MatrixXd const& rho_r,
                             Eigen::MatrixXd const& vx_l,
                             Eigen::MatrixXd const& vx_r,
                             Eigen::MatrixXd const& vy_l,
                             Eigen::MatrixXd const& vy_r,
                             Eigen::MatrixXd const& p_l,
                             Eigen::MatrixXd const& p_r)
{
  auto fluxes = ConservedQuantities{};

  auto en_l = p_l / (ic::gamma - 1.) +
              0.5 * rho_l.cwiseProduct(vx_l.cwiseProduct(vx_l) +
                                       vy_l.cwiseProduct(vy_l));

  auto en_r = p_r / (ic::gamma - 1.) +
              0.5 * rho_r.cwiseProduct(vx_r.cwiseProduct(vx_r) +
                                       vy_r.cwiseProduct(vy_r));

  auto rho_star = 0.5 * (rho_l + rho_r);

  auto momx_star = 0.5 * (rho_l.cwiseProduct(vx_l) + rho_r.cwiseProduct(vx_r));

  auto momy_star = 0.5 * (rho_l.cwiseProduct(vy_l) + rho_r.cwiseProduct(vy_r));

  auto en_star = 0.5 * (en_l + en_r);

  auto p_star =
      (ic::gamma - 1) * (en_star - 0.5 * (momx_star.cwiseProduct(momx_star) +
                                          momy_star.cwiseProduct(momy_star))
                                             .cwiseQuotient(rho_star));

  fluxes.mass = momx_star;
  fluxes.momentum_x =
      (momx_star.cwiseProduct(momx_star)).cwiseQuotient(rho_star) + p_star;
  fluxes.momentum_y =
      (momx_star.cwiseProduct(momy_star)).cwiseQuotient(rho_star);
  fluxes.energy =
      ((en_star + p_star).cwiseProduct(momx_star)).cwiseQuotient(rho_star);

  auto c_l =
      (ic::gamma * (p_l.cwiseQuotient(rho_l))).cwiseSqrt() + vx_l.cwiseAbs();

  auto c_r =
      (ic::gamma * (p_r.cwiseQuotient(rho_r))).cwiseSqrt() + vx_r.cwiseAbs();

  auto c = c_l.cwiseMax(c_r);

  fluxes.mass -= 0.5 * c.cwiseProduct(rho_l - rho_r);
  fluxes.momentum_x -=
      0.5 *
      c.cwiseProduct((rho_l.cwiseProduct(vx_l) - rho_r.cwiseProduct(vx_r)));
  fluxes.momentum_y -=
      0.5 * c.cwiseProduct(rho_l.cwiseProduct(vy_l) - rho_r.cwiseProduct(vy_r));
  fluxes.energy -= 0.5 * c.cwiseProduct(en_l - en_r);

  return fluxes;
}

Eigen::MatrixXd apply_fluxes(Eigen::MatrixXd& f,
                             Eigen::MatrixXd const& flux_f_x,
                             Eigen::MatrixXd const& flux_f_y,
                             double dt)
{
  f += -dt * ic::dx * flux_f_x;

  Eigen::MatrixXd flux_f_x_shiftl(ic::n_x, ic::n_y + 2);
  flux_f_x_shiftl << flux_f_x.bottomRows(1), flux_f_x.topRows(ic::n_x - 1);
  f += dt * ic::dx * flux_f_x_shiftl;

  f += -dt * ic::dx * flux_f_y;

  Eigen::MatrixXd flux_f_y_shiftl(ic::n_x, ic::n_y + 2);
  flux_f_y_shiftl << flux_f_y.rightCols(1), flux_f_y.leftCols(ic::n_y + 1);
  f += dt * ic::dx * flux_f_y_shiftl;

  return f;
}

void print_matrix(Eigen::MatrixXd const& m)
{
  for (uint32_t i{0}; i != ic::n_x; ++i) {
    for (uint32_t j{0}; j != ic::n_y + 2; ++j) {
      printf("%12.8e ", m(i, j));
    }
    std::cout << '\n';
  }
  std::cout << "\n\n";
}