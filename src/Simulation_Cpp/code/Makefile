CC = g++-13 -O3 -std=c++17
CFLAGS = -Wall -g -fopenmp -fopenmp-simd

LE_1D_confine: LE_1D_confine.o print_file.o periodic_boundary_conditions.o initialization.o update_position.o check_nooverlap.o
	$(CC) $(CFLAGS) -o LE_1D_confine.out LE_1D_confine.o print_file.o periodic_boundary_conditions.o initialization.o update_position.o check_nooverlap.o

LE_1D_confine.o: LE_1D_confine.cpp
	$(CC) $(CFLAGS) -c LE_1D_confine.cpp

print_file.o: print_file.cpp
	$(CC) -c print_file.cpp

periodic_boundary_conditions.o: periodic_boundary_conditions.cpp
	$(CC) $(CFLAGS) -c periodic_boundary_conditions.cpp

initialization.o: initialization.cpp
	$(CC) $(CFLAGS) -c initialization.cpp

update_position.o: update_position.cpp
	$(CC) $(CFLAGS) -c update_position.cpp

check_nooverlap.o: check_nooverlap.cpp
	$(CC) $(CFLAGS) -c check_nooverlap.cpp

clean:
	rm *.o

	