// Copyright 2024 Jeremy Vachier
#pragma once
#include <random>

void initialization(
  double *x, int Particles,
  std::default_random_engine &generator,
  std::uniform_real_distribution<double> &distribution);
