MPICC = mpicc
CC = clang

CFLAGS = -O3 -std=c11 -Wall -Wextra

OMP_PREFIX = /opt/homebrew/opt/libomp

OMP_FLAGS = -Xpreprocessor -fopenmp \
            -I$(OMP_PREFIX)/include

OMP_LIBS = -L$(OMP_PREFIX)/lib -lomp \
           -Wl,-rpath,$(OMP_PREFIX)/lib

.PHONY: all clean

all: bin/mpi_matmul bin/run_parallel

bin:
	mkdir -p bin

bin/mpi_matmul: src/mpi_matmul.c | bin
	$(MPICC) $(CFLAGS) $< -o $@

bin/run_parallel: src/matrix_parallel.c | bin
	$(CC) $(CFLAGS) $(OMP_FLAGS) $< $(OMP_LIBS) -o $@

clean:
	rm -f bin/mpi_matmul bin/run_parallel