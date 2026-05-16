// Copyright 2024 Jeremy Vachier
#include "headers/periodic_boundary_conditions.h"

void periodic_boundary_conditions(
  double *x, int Particles,
  double Wall) {
  // Ternary form is branchless — the compiler emits masked SIMD selects
  // (VBLENDVPD on AVX, BFSEL on NEON) instead of branch mispredictions.
#pragma omp parallel for simd
  for (int k = 0; k < Particles; k++) {
    double v = x[k];
    x[k] = (v > Wall) ? -Wall : (v < -Wall ? Wall : v);
  }
}
