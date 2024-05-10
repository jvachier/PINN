#include "headers/periodic_boundary_conditions.h"

using namespace std;

void periodic_boundary_conditions(
	double *x, int Particles,
	double Wall)
{
#pragma omp parallel for simd
	for (int k = 0; k < Particles; k++)
	{
		if (x[k] > Wall ){
			x[k] = -Wall;
		}
		if (x[k] < Wall ){
			x[k] = Wall;
		}
	}
}