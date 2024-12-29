#include "headers/print_file.h"

using namespace std;

// void print_file(
//   double *x,
//   int Particles, int time,
//   FILE *datacsv) {
//   for (int k = 0; k < Particles; k++) {
//     fprintf(datacsv, "Particles%06d,%lf,%d\n", k, x[k], time);
//   }
// }

void print_file(
  double *x,
  int Particles, int time,
  FILE *datacsv) {
  fprintf(datacsv, "%d,", time);
  for (int k = 0; k < Particles; k++) {
    fprintf(datacsv, "%lf,", x[k]);
  }
  fprintf(datacsv, "\n");
}

void print_file_binary(
  double *x,
  int Particles, int time,
  FILE *datacsv) {
  fwrite(&time, sizeof(int), 1, datacsv);
  for (int k = 0; k < Particles; k++) {
    fwrite(&x[k], sizeof(double), 1, datacsv);
  }
}
