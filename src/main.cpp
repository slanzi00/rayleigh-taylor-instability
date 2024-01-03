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

int main()
{
  auto axis = Axis{ic::x_lim, ic::y_lim};
  auto primitives = PrimitiveQuantities{axis};
  auto conserved = get_conserved(primitives);
  primitives = get_primitive(conserved);
  auto dt = get_dt(primitives);
  // whilw loop here
  // add source (half-step)
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

  auto v_x_prime =
      primitives.v_x - 0.5 * dt *
                           (primitives.v_x.cwiseProduct(grad_v_x.first) +
                            primitives.v_y.cwiseProduct(grad_v_x.second) +
                            grad_pressure.first.cwiseQuotient(primitives.rho));

  auto v_y_prime =
      primitives.v_y - 0.5 * dt *
                           (primitives.v_x.cwiseProduct(grad_v_y.first) +
                            primitives.v_y.cwiseProduct(grad_v_y.second) +
                            grad_pressure.second.cwiseQuotient(primitives.rho));

  auto pressure_prime = primitives.pressure -
                        0.5 * dt *
                            (ic::gamma * primitives.pressure.cwiseProduct(
                                             grad_v_x.first + grad_v_y.second) +
                             primitives.v_x.cwiseProduct(grad_pressure.first) +
                             primitives.v_y.cwiseProduct(grad_pressure.second));

  // std::cout << primitives.rho << "\n\n"
  //           << primitives.v_x << "\n\n"
  //           << primitives.v_y << "\n\n"
  //           << primitives.pressure << '\n';

  // std::cout << conserved.mass << "\n\n"
  //           << conserved.momentum_x << "\n\n"
  //           << conserved.momentum_y << "\n\n"
  //           << conserved.energy << '\n';
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
  auto fill_m_dt = [&](uint8_t i, uint8_t j) {
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