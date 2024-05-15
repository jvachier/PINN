#include "headers/print_file.h"

using namespace std;

void print_file(
  double *x,
  int Particles, int time,
  FILE *datacsv) {
  for (int k = 0; k < Particles; k++) {
    fprintf(datacsv, "Particles%06d,%lf,%d\n", k, x[k], time);
  }
}
