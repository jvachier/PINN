#pragma once
#include <omp.h>
#include <random>
#include <vector>

// Each element of 'generators' is private to one OpenMP thread.
// This eliminates data races on the RNG state while keeping full parallelism.
// The arithmetic x[k] += drift + noise*prefactor is auto-vectorised by the
// compiler under -O3 -march=native; no #pragma omp simd is needed.
void update_position(
  double *x, int Particles,
  double delta,
  double vs, double prefactor_xi_px,
  std::vector<std::default_random_engine> &generators);
