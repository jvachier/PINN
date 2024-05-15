#include "headers/update_position.h"

using namespace std;

void update_position(
  double *x, int Particles,
  double delta, double xi_px,
  double vs, double prefactor_xi_px,
  default_random_engine generator,
  normal_distribution<double> Gaussdistribution) {
// Second position
#pragma omp parallel for simd
  for (int k = 0; k < Particles; k++) {
    xi_px = Gaussdistribution(generator);
    x[k] = x[k] + vs * delta + xi_px * prefactor_xi_px;
  }
}
