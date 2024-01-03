#ifndef TYPES_HPP
#define TYPES_HPP

#include <Eigen/Core>

#include "constants.hpp"

using MatrixPair = std::pair<Eigen::MatrixXd, Eigen::MatrixXd>;

struct Axis
{
  Eigen::ArrayXd x;
  Eigen::ArrayXd y;

  Axis(double, double);
};

struct PrimitiveQuantities
{
  Eigen::MatrixXd rho;
  Eigen::MatrixXd v_x;
  Eigen::MatrixXd v_y;
  Eigen::MatrixXd pressure;

  PrimitiveQuantities();
  PrimitiveQuantities(Axis const&);
};

struct ConservedQuantities
{
  Eigen::MatrixXd mass;
  Eigen::MatrixXd momentum_x;
  Eigen::MatrixXd momentum_y;
  Eigen::MatrixXd energy;
};

struct SpatialExtrapolated
{
  Eigen::MatrixXd xl;
  Eigen::MatrixXd xr;
  Eigen::MatrixXd yl;
  Eigen::MatrixXd yr;
};

#endif