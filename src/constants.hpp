#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

namespace ic {
constexpr uint32_t n_x = 256;
constexpr uint32_t n_y = n_x * 3;
constexpr double x_lim = 0.5;
constexpr double y_lim = 1.5;
constexpr double dx = x_lim / n_x;  // dy = dx
constexpr double volume = dx * dx;
constexpr double p_0 = 2.5;
constexpr double g = -0.1;
constexpr double gamma = 1.4;
constexpr double t_end = 30.;
}  // namespace ic

#endif