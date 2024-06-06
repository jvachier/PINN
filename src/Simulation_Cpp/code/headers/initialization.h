#include <time.h>
#include <stdio.h>
#include <omp.h>  // import library to use pragma
#include <iostream>
#include <random>
#include <cstring>
#include <cmath>

void initialization(
  double *x, int Particles,
  std::default_random_engine &generator,
  std::uniform_real_distribution<double> &distribution);
