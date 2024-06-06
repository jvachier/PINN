/*
 * Author: Jeremy Vachier - Physics Informed Neural Networks
 * Purpose: Langevin Equation 1D using an Euler-Mayurama algorithm
 * Language: C++
 * Date: 2024
 * Compilation line to use pragma: g++ name.cpp -fopenmp -o name.o (on mac run g++-13 ; 13 latest version obtain using brew list gcc)
 * Compilation line to use pragma, simd (vectorization) and tuple: g++ -O3 -std=c++17 name.cpp -fopenmp -o name.o
 */

#include <time.h>
#include <stdio.h>
#include <omp.h>  // import library to use pragma
#include <iostream>
#include <random>
#include <cstring>
#include <cmath>

#include "headers/print_file.h"
#include "headers/periodic_boundary_conditions.h"
#include "headers/initialization.h"
#include "headers/update_position.h"
#include "headers/check_nooverlap.h"

#define PI 3.141592653589793
#define N_thread 6

using namespace std;

int main(int argc, char *argv[]) {
  // File
  FILE *datacsv;
  FILE *parameter;
  parameter = fopen("parameter.txt", "r");
  datacsv = fopen("../data/simulation.csv", "w");

  // check if the file parameter is exist
  if (parameter == NULL) {
    printf("no such file.");
    return 0;
  }

  omp_set_num_threads(N_thread);

  // read the parameters from the file
  double delta, Dt, vs;
  double Wall;
  int Particles;
  int N, timestep;  // number of iterations

  fscanf(parameter, "%lf\t%d\t%lf\t%lf\t%lf\t%d\t%d\n", \
    &delta, &Particles, &Dt, &vs, &Wall, &N, &timestep);
  printf("%lf\t%d\t%lf\t%lf\t%lf\t%d\t%d\n", \
    delta, Particles, Dt, vs, Wall, N, timestep);

  // Position
  double *x = reinterpret_cast<double*> \
    (malloc(Particles * sizeof(double)));  // x-position

  // parameters
  // const int L = 1.0; // particle size

  // initialization of the random generator
  random_device rdev;
  default_random_engine generator(rdev());  // random seed -> rdev

  // Distributions Gaussian
  normal_distribution<double> Gaussdistribution(0.0, 1.0);
  // Distribution Uniform for initialization
  uniform_real_distribution<double> distribution(-Wall, Wall);

  double xi_px = 0.0;  // noise for x-position

  // double phi = 0.0;
  double prefactor_xi_px = sqrt(2.0 * delta * Dt);

  // Open MP to get execution time
  double itime, ftime, exec_time;
  itime = omp_get_wtime();

  fprintf(datacsv, "Particles,x-position,time\n");

// initialization position and activity
  initialization(
    x, Particles,
    generator, distribution);

  // check_nooverlap(
  //   x, Particles, L,
  //   generator, distribution);
  int time = 0;
  print_file(
    x,
    Particles, time,
    datacsv);
  printf("Initialization done.\n");

  // Time evoultion
  for (int time = 0; time < N; time++) {
    update_position(
      x, Particles,
      delta, xi_px,
      vs, prefactor_xi_px,
      generator, Gaussdistribution);
  // periodic_boundary_conditions(
  //  x, Particles,
  //  Wall);

    if (time % timestep == 0 && time > 0) {
      print_file(
        x,
        Particles, time,
        datacsv);
    }
  }

  ftime = omp_get_wtime();
  exec_time = ftime - itime;
  printf("Time taken is %f", exec_time);

  free(x);

  fclose(datacsv);
  return 0;
}
