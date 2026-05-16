// Copyright 2024 Jeremy Vachier
// Unit tests for the C++ simulation core.
//
// Analytically-known properties tested:
//   t=0  : initialization sets every particle to x=0
//   t=dt : mean(x) = vs*delta,  var(x) = 2*Dt*delta  (Euler-Maruyama)
//   pbc  : periodic_boundary_conditions clamps to [-Wall, +Wall]

#include <omp.h>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "../headers/initialization.h"
#include "../headers/periodic_boundary_conditions.h"
#include "../headers/update_position.h"

// ── Minimal test harness -----------------------------------------------------
static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_TRUE(cond, msg)                                              \
  do {                                                                      \
    if (cond) {                                                             \
      printf("  PASS  %s\n", (msg)); ++g_pass;                             \
    } else {                                                                \
      printf("  FAIL  %s  (%s:%d)\n",  \
             (msg), __FILE__, __LINE__); \
      ++g_fail;                                                             \
    }                                                                       \
  } while (0)

#define ASSERT_NEAR(a, b, tol, msg) \
  ASSERT_TRUE(std::fabs((a) - (b)) < (tol), (msg))

// ── Helpers ------------------------------------------------------------------
static std::vector<std::default_random_engine> make_generators(unsigned seed) {
  int nt = omp_get_max_threads();
  std::vector<std::default_random_engine> gens(nt);
  std::mt19937 seeder(seed);
  for (int t = 0; t < nt; ++t) gens[t].seed(seeder());
  return gens;
}

// --- Test: initialization ---------------------------------------------------
static void test_initialization() {
  printf("\n[initialization]\n");

  const int N = 1000;
  std::vector<double> x(N, 99.0);  // non-zero sentinel

  std::default_random_engine gen(42);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  initialization(x.data(), N, gen, dist);

  bool all_zero = true;
  for (int k = 0; k < N; ++k) {
    if (x[k] != 0.0) {
      all_zero = false;
      break;
    }
  }
  ASSERT_TRUE(all_zero, "all positions are 0.0 after initialization");
}

// --- Test: update_position statistics after one step -------------------------
//
// Starting from x[k]=0, after one Euler-Maruyama step:
//   x[k] = drift + N(0,1)*prefactor
// where drift = vs*delta, prefactor = sqrt(2*Dt*delta).
//
// By the law of large numbers (N=50000):
//   E[mean(x)] = drift,   std_of_mean ≈ prefactor/sqrt(N) ≈ 0.0045
//   E[var(x)]  = prefactor^2,  tolerance 0.05 is >10 sigma.
static void test_update_position_statistics() {
  printf("\n[update_position — Euler-Maruyama statistics after one step]\n");

  const int    N          = 50000;
  const double delta      = 1.0;
  const double vs         = 1.0;
  const double Dt         = 0.5;
  const double prefactor  = std::sqrt(2.0 * Dt * delta);  // = 1.0

  std::vector<double> x(N, 0.0);
  auto gens = make_generators(12345);

  update_position(x.data(), N, delta, vs, prefactor, gens);

  const double mean = std::accumulate(x.begin(), x.end(), 0.0) / N;
  double var = 0.0;
  for (double v : x) var += (v - mean) * (v - mean);
  var /= (N - 1);

  ASSERT_NEAR(mean, vs * delta,       0.05, "mean(x) ~ vs*delta");
  ASSERT_NEAR(var,  2.0 * Dt * delta, 0.05, "var(x)  ~ 2*Dt*delta");
}

// --- Test: drift accumulates over multiple steps -----------------------------
//
// After T steps from x=0: E[mean(x)] = T * vs*delta.
static void test_update_position_drift_accumulation() {
  printf("\n[update_position — drift accumulation over T steps]\n");

  const int    N         = 50000;
  const double delta     = 1.0;
  const double vs        = 2.0;
  const double Dt        = 0.5;
  const double prefactor = std::sqrt(2.0 * Dt * delta);
  const int    T         = 10;

  std::vector<double> x(N, 0.0);
  auto gens = make_generators(99999);

  for (int t = 0; t < T; ++t)
    update_position(x.data(), N, delta, vs, prefactor, gens);

  const double mean = std::accumulate(x.begin(), x.end(), 0.0) / N;
  // tolerance: 5*sqrt(T)*prefactor/sqrt(N) ~ 5*3.16*1/223 ~ 0.071
  ASSERT_NEAR(mean, T * vs * delta, 0.15, "mean(x) ~ T*vs*delta after T steps");
}

// --- Test: periodic_boundary_conditions --------------------------------------
static void test_periodic_boundary_conditions() {
  printf("\n[periodic_boundary_conditions]\n");

  const double Wall = 2.0;
  double x[] = {2.5, -3.0, 1.0, 2.0, -2.0};
  const int N = 5;

  periodic_boundary_conditions(x, N, Wall);

  ASSERT_NEAR(x[0], -Wall, 1e-12, "x >  Wall  →  −Wall");
  ASSERT_NEAR(x[1],  Wall, 1e-12, "x < −Wall  →  +Wall");
  ASSERT_NEAR(x[2],  1.0,  1e-12, "|x| < Wall →  unchanged");
  ASSERT_NEAR(x[3],  2.0,  1e-12, "x == Wall  →  unchanged (not strictly >)");
  ASSERT_NEAR(x[4], -2.0, 1e-12,
              "x == -Wall -> unchanged (not strictly <)");
}

// ── Entry point --------------------------------------------------------------
int main() {
  printf("=== C++ simulation unit tests ===\n");

  test_initialization();
  test_update_position_statistics();
  test_update_position_drift_accumulation();
  test_periodic_boundary_conditions();

  printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return (g_fail > 0) ? 1 : 0;
}
