# Assignment 4 — Distributed Matrix Multiplication (MPI)

This starter implements dense matrix multiplication with MPI using row-wise decomposition.

## Design

- Matrix A is partitioned by rows across MPI processes.
- Matrix B is broadcast in full to every process.
- Each process computes its assigned rows of C using an i-k-j loop order.
- The final matrix C is collected on rank 0 with `MPI_Gatherv`.
- `MPI_Scatterv` / `MPI_Gatherv` support matrix sizes that are not divisible by the number of workers.
- The program reports setup, broadcast, scatter, computation, gather, communication and end-to-end distributed time.
- Correctness is represented by a checksum; for the report, validate small cases against the sequential implementation as well.

## Install Open MPI on macOS

```bash
brew install open-mpi
```

Verify:

```bash
mpicc --version
mpirun --version
```

## Build

```bash
make
```

## Run one configuration

```bash
mpirun -np 4 ./mpi_matmul 1000 12345
```

## Benchmark

Default configuration: sizes 500, 1000, 1500 and 2000; workers 1, 2, 4, 8 and 10; one warm-up and five measured repetitions.

```bash
./run_benchmarks.sh
python3 analyze.py
```

To change the experiment:

```bash
REPS=5 SIZES="1000 1500 2000 2500" WORKERS="1 2 4 8 10" ./run_benchmarks.sh
```

## Important methodological note

Running several MPI processes on a single Mac is a local multi-process experiment. It can measure framework/process overhead and local strong scaling, but it does not demonstrate inter-machine network performance. State this explicitly in the report unless the experiment is run on multiple physical machines.
