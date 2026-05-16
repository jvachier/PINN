# On macOS use Homebrew clang (compatible with macOS SDK) + Homebrew libomp.
# On Linux the Docker image uses g++-14; this Makefile is for local macOS builds.
CC      = clang++
LIBOMP  = /opt/homebrew/opt/libomp
# -O3: full optimisation  -march=native: NEON / AVX auto-vectorisation
# -Xpreprocessor -fopenmp: clang needs this to enable OpenMP on macOS
CFLAGS  = -O3 -std=c++17 -march=native -Wall -g \
          -Xpreprocessor -fopenmp \
          -I$(LIBOMP)/include
LDFLAGS = -L$(LIBOMP)/lib -lomp

OBJS = LE_1D_confine.o print_file.o periodic_boundary_conditions.o \
       initialization.o update_position.o check_nooverlap.o

# All objects except the main translation unit (has its own main())
TEST_OBJS = print_file.o periodic_boundary_conditions.o \
            initialization.o update_position.o check_nooverlap.o

.PHONY: all clean run test

all: LE_1D_confine.out

LE_1D_confine.out: $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

run: LE_1D_confine.out
	./LE_1D_confine.out

test: test_simulation.out
	./test_simulation.out

test_simulation.out: tests/test_simulation.cpp $(TEST_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -I. -o $@ tests/test_simulation.cpp $(TEST_OBJS)

clean:
	rm -f *.o LE_1D_confine.out test_simulation.out
