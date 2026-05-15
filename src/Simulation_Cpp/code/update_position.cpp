#include "headers/update_position.h"

void update_position(
  double *x, int Particles,
  double delta,
  double vs, double prefactor_xi_px,
  std::vector<std::default_random_engine> &generators) {

  const double drift = vs * delta;
  // Each thread uses its own generator — no race condition.
  // -O3 -march=native auto-vectorises the fused multiply-add arithmetic.
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    std::normal_distribution<double> dist(0.0, 1.0);
#pragma omp for
    for (int k = 0; k < Particles; k++) {
      x[k] += drift + dist(generators[tid]) * prefactor_xi_px;
    }
  }
}
