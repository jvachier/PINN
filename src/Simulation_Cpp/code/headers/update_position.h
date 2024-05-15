#include <iostream>
#include <random>
#include <cstring>
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <omp.h>  // import library to use pragma

void update_position(
  double *x,  int Particles,
  double delta, double xi_px,
  double vs, double prefactor_xi_px,
  std::default_random_engine *generator,
  std::normal_distribution<double> *Gaussdistribution);
