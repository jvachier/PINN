#include "headers/initialization.h"

using namespace std;

void initialization(
  double *x, int Particles,
  default_random_engine *generator,
  uniform_real_distribution<double> *distribution) {
  // Position
#pragma omp parallel for simd
  for (int k = 0; k < Particles; k++) {
  // x[k] = distribution(generator);
  x[k] = 0.0;
  }
}
