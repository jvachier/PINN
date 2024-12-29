#include <time.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <cstring>
#include <cmath>

void print_file(
  double *x,
  int Particles, int time,
  FILE *datacsv);

void print_file_binary(
  double *x,
  int Particles, int time,
  FILE *datacsv);
