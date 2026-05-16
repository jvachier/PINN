// Copyright 2024 Jeremy Vachier
/*
 * Author: Jeremy Vachier - Physics Informed Neural Networks
 * Purpose: Langevin Equation 1D using an Euler-Mayurama algorithm
 * Language: C++
 * Date: 2024
 * Compilation line to use pragma: g++ name.cpp -fopenmp -o name.o (on mac run g++-13 ; 13 latest version obtain using brew list gcc)
 * Compilation line to use pragma, simd (vectorization) and tuple: g++ -O3 -std=c++17 name.cpp -fopenmp -o name.o
 */

#include <omp.h>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "headers/print_file.h"
#include "headers/periodic_boundary_conditions.h"
#include "headers/initialization.h"
#include "headers/update_position.h"
#include "headers/check_nooverlap.h"

#define PI 3.141592653589793
#define Type_Bin true

int main(int argc, char *argv[]) {
  // File
  FILE *datacsv;
  FILE *parameter;
  parameter = fopen("parameter.txt", "r");

  if (Type_Bin == true) {
    datacsv = fopen("../data/simulation.bin", "wb");
  } else {
    datacsv = fopen("../data/simulation.csv", "w");
  }

  if (parameter == NULL) {
    printf("no such file.\n");
    fclose(datacsv);
    return 1;
  }

  // read the parameters from the file
  double delta, Dt, vs;
  double Wall;
  int Particles;
  int total_time, timestep;  // number of iterations

  fscanf(parameter, "%lf\t%d\t%lf\t%lf\t%lf\t%d\t%d\n", \
    &delta, &Particles, &Dt, &vs, &Wall, &total_time, &timestep);
  printf("%lf\t%d\t%lf\t%lf\t%lf\t%d\t%d\n", \
    delta, Particles, Dt, vs, Wall, total_time, timestep);

  // Position
  double *x = reinterpret_cast<double*> \
    (malloc(Particles * sizeof(double)));  // x-position

  // parameters
  // const int L = 1.0; // particle size

  // initialization of the random generator
  std::random_device rdev;
  std::default_random_engine generator(rdev());

  // Per-thread generators — each seeded independently to avoid correlations.
  int N_thread = omp_get_max_threads();
  std::vector<std::default_random_engine> generators(N_thread);
  for (int t = 0; t < N_thread; t++) {
    generators[t].seed(rdev());
  }

  // Distributions Gaussian
  std::normal_distribution<double> Gaussdistribution(0.0, 1.0);
  // Distribution Uniform for initialization
  std::uniform_real_distribution<double> distribution(-Wall, Wall);

  double prefactor_xi_px = sqrt(2.0 * delta * Dt);

  // Open MP to get execution time
  double itime = omp_get_wtime();

  if (Type_Bin == false) {
    // fprintf(datacsv, "Particles,x-position,time\n");
    fprintf(datacsv, "time,");
    for (int i = 0 ; i < Particles ; i++) {
      fprintf(datacsv, "Particles%06d,", i);
    }
    fprintf(datacsv, "\n");
  }

// initialization position and activity
  initialization(
    x, Particles,
    generator, distribution);

  // check_nooverlap(
  //   x, Particles, L,
  //   generator, distribution);
  int time = 0;
  if (Type_Bin == true) {
    print_file_binary(
      x,
      Particles, time,
      datacsv);
  } else {
    print_file(
      x,
      Particles, time,
      datacsv);
  }
  printf("Initialization done.\n");

  // Time evoultion
  for (int time = 0; time < total_time; time++) {
    update_position(
      x, Particles,
      delta,
      vs, prefactor_xi_px,
      generators);
  // periodic_boundary_conditions(
  //  x, Particles,
  //  Wall);

    if (time % timestep == 0 && time > 0) {
      if (Type_Bin == true) {
        print_file_binary(
          x,
          Particles, time,
          datacsv);
      } else {
        print_file(
          x,
          Particles, time,
          datacsv);
      }
    }
  }

  double exec_time = omp_get_wtime() - itime;
  printf("Time taken is %f\n", exec_time);

  free(x);

  fclose(datacsv);
  return 0;
}
