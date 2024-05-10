#include <iostream>
#include <random>
#include <cstring>
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <omp.h> //import library to use pragma

void periodic_boundary_conditions(
	double *x, int Particles,
	double Wall
);