// Copyright 2024 Jeremy Vachier
#include "headers/initialization.h"

void initialization(
  double *x, int Particles,
  std::default_random_engine &generator,
  std::uniform_real_distribution<double> &distribution) {
  // Position
#pragma omp parallel for simd
  for (int k = 0; k < Particles; k++) {
  // x[k] = distribution(generator);
  x[k] = 0.0;
  }
}
