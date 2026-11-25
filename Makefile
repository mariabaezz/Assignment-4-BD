MPICC ?= mpicc
CC = clang
OMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null)
CFLAGS = -O3 -std=c11 -Wall -Wextra
all: mpi_matmul run_parallel
mpi_matmul: mpi_matmul.c
	$(MPICC) $(CFLAGS) $< -o $@
run_parallel: matrix_parallel.c
	$(CC) $(CFLAGS) -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include $< -L$(OMP_PREFIX)/lib -lomp -Wl,-rpath,$(OMP_PREFIX)/lib -o $@
clean:
	rm -f mpi_matmul run_parallel
