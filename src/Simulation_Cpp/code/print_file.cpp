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
    fprintf(datacsv, "%.3lf,", x[k]);
  }
  fprintf(datacsv, "\n");
}

void print_file_binary(
  double *x,
  int Particles, int time,
  FILE *datacsv) {
  fwrite(&time, sizeof(int), 1, datacsv);
  // Cast to float (4 bytes) — ample for 3 d.p. precision, halves file size.
  for (int k = 0; k < Particles; k++) {
    float xf = static_cast<float>(x[k]);
    fwrite(&xf, sizeof(float), 1, datacsv);
  }
}
