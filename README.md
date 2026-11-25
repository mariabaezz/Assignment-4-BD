# Assignment 4 – Distributed Matrix Multiplication

**Big Data · Academic Year 2026–2027**  
Degree in Data Science and Engineering  
Universidad de Las Palmas de Gran Canaria

## 1. Overview

This project investigates the performance and scalability of dense matrix multiplication using the Message Passing Interface (MPI). The distributed implementation is compared with sequential and OpenMP implementations adapted from Assignment 3.

The main objective is to evaluate how increasing the number of MPI processes affects execution time, speedup, efficiency, communication overhead and memory consumption.

All experiments were conducted locally on a single Apple M4 machine. Therefore, the results characterise local multi-process execution rather than inter-machine network performance.

## 2. Research Question

How does increasing the number of MPI processes affect the performance and resource consumption of dense matrix multiplication, and how does MPI compare with sequential and OpenMP implementations under equivalent input and computational conditions?

The initial hypothesis is that increasing the number of processes reduces computation time, but communication, synchronisation, framework overhead and memory replication limit scalability.

## 3. Implementations

| Method | Description |
|---|---|
| Sequential | Single-process C implementation using reordered i-k-j loops. |
| OpenMP | Shared-memory parallel implementation with static row scheduling. |
| MPI | Multi-process implementation using row-based matrix decomposition and collective communication. |

All implementations use:

- Dense square matrices.
- 64-bit signed integer elements.
- Row-major storage.
- Reordered i-k-j multiplication.
- Identical deterministic input generation.
- Input values between 0 and 9.

The input generator is a 32-bit linear congruential generator (LCG), with seeds 42 and 4242 for matrices A and B, respectively.

### MPI architecture

The MPI implementation distributes the computation using the following operations:

1. Rank 0 generates the input matrices.
2. `MPI_Bcast` distributes matrix B to every process.
3. `MPI_Scatterv` distributes rows of matrix A.
4. Each process computes its assigned rows of matrix C.
5. `MPI_Gatherv` collects the partial results on rank 0.
6. The final checksum is calculated.

Variable-sized collective operations allow the implementation to handle matrix dimensions that are not divisible by the number of processes.

Each process stores a complete copy of matrix B. Consequently, the implementation introduces additional memory consumption compared with shared-memory OpenMP execution.

## 4. Repository Structure

```text
Assignment-4-BD/
├── src/
│   ├── matrix_parallel.c
│   └── mpi_matmul.c
├── scripts/
│   ├── benchmark_final.py
│   └── plots.py
├── tests/
│   └── test_correctness.py
├── results/
│   ├── results_final.csv
│   └── summary_final.csv
├── preliminary/
│   ├── analyze.py
│   ├── run_benchmarks.sh
│   ├── results_mpi.csv
│   └── summary_mpi.csv
├── figures/
├── report/
├── Makefile
├── requirements.txt
├── .gitignore
└── README.md
```

The `preliminary/` directory preserves the initial exploratory MPI experiments. These measurements used a different input generator and are not included in the final comparative analysis.

Compiled executables are generated in `bin/` and excluded from version control.

## 5. Experimental Environment

| Component | Specification |
|---|---|
| Computer | MacBook Pro |
| Processor | Apple M4 |
| Architecture | ARM64 |
| Physical cores | 10 |
| Logical processors | 10 |
| RAM | 16 GB |
| Operating system | macOS |
| Compiler | Apple Clang 21.0.0 |
| MPI framework | Open MPI 5.0.11 |
| OpenMP runtime | libomp |
| Compiler optimisation | -O3 |
| Programming language | C11 |
| Analysis | Python and Matplotlib |

The Apple M4 uses a heterogeneous CPU architecture. Consequently, the available CPU cores should not be assumed to provide identical computational performance.

## 6. Installation and Compilation

### Requirements

- macOS with Apple Silicon.
- Homebrew.
- Open MPI.
- OpenMP runtime.
- Python 3.
- Matplotlib.

Install the required native dependencies:

```bash
brew install open-mpi libomp
```

Install the Python dependencies using a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Clone the repository:

```bash
git clone https://github.com/mariabaezz/Assignment-4-BD.git
cd Assignment-4-BD
```

The virtual environment should be created from the repository root.

Compile both implementations:

```bash
make
```

To rebuild from scratch:

```bash
make clean
make
```

## 7. Execution

### Sequential

```bash
./bin/run_parallel seq 2000 1
```

### OpenMP

```bash
./bin/run_parallel omp 2000 8
```

### MPI

```bash
mpirun -np 8 ./bin/mpi_matmul 2000
```

The programs report execution times, checksum and peak resident memory measurements. The MPI implementation additionally reports setup, broadcast, scatter, computation and gathering durations.

## 8. Correctness Validation

Correctness is assessed through an independent Python reference implementation using the same deterministic input generator.

Run the validation suite:

```bash
python3 tests/test_correctness.py
```

The suite covers:

- Seven matrix sizes: 1, 2, 3, 7, 11, 32 and 100.
- Sequential, OpenMP and MPI execution.
- Multiple thread and process configurations.
- Uneven row partitioning.
- Configurations with more MPI processes than matrix rows.
- Comparison against independently calculated reference checksums.

The suite comprises 77 checksum comparisons.

Additionally, all 275 recorded benchmark measurements produce matching checksums for each matrix size.

Checksum agreement provides evidence of consistency but does not constitute a complete element-wise correctness proof.

## 9. Benchmarking Methodology

The final experimental configurations are:

| Parameter | Values |
|---|---|
| Matrix sizes | 500, 1000, 1500, 2000, 3000 |
| Sequential resources | 1 |
| OpenMP threads | 1, 2, 4, 8, 10 |
| MPI processes | 1, 2, 4, 8, 10 |
| Warm-up executions | 1 per configuration |
| Measured repetitions | 5 per configuration |
| Total configurations | 55 |
| Total measured executions | 275 |
| Execution timeout | 120 seconds |
| Configuration randomisation seed | 42 |

Configuration order is randomised within each matrix size and repetition.

Warm-up executions are excluded from the reported measurements. Results are summarised using the median and interquartile range (IQR).

### Run the benchmarks

```bash
python3 scripts/benchmark_final.py
```

**Warning:** this command regenerates and overwrites `results/results_final.csv`. The original experimental measurements are already included in the repository.

### Generate statistics and figures

```bash
python3 scripts/plots.py
```

This command generates the statistical summary and all performance figures from the recorded measurements.

## 10. Performance Metrics

### Computation time

Measures the multiplication kernel. For MPI, the maximum computation duration across processes is reported.

### Operation time

For MPI, includes matrix distribution, local computation and result collection. For the sequential and OpenMP implementations, it corresponds to the measured multiplication kernel.

These operation boundaries are therefore not strictly identical across methods.

### End-to-end execution time

Measured externally using Python's monotonic `time.perf_counter()` timer.

Includes process launch, input generation, computation, result consumption and process termination. MPI measurements also include launcher and framework startup costs.

End-to-end time is used as the primary metric for cross-method comparison.

### Speedup and efficiency

Speedup is calculated as:

```text
S(p) = T(1) / T(p)
```

Efficiency is calculated as:

```text
E(p) = S(p) / p
```

Two speedup references are considered:

- The sequential implementation.
- The one-worker configuration of the same implementation.

MPI strong scaling is evaluated by keeping the matrix size fixed while increasing the number of processes.

### Memory consumption

Peak resident set size (RSS) is measured using `getrusage()`.

For MPI, two metrics are recorded:

- Maximum individual process peak RSS.
- Sum of individual process peak RSS values.

The sum of individual peaks is not equivalent to the simultaneous physical memory peak of the entire application. Shared runtime pages may also be counted more than once.

### Communication timings

MPI collective operation durations are measured separately.

Since phase maxima may originate from different processes and may include synchronisation delays, these values are not treated as additive components of the overall execution time.

## 11. Results

The final experiment completed all 275 measured executions successfully.

The largest tested matrix size was 3000 × 3000.

The main observations are:

- MPI reduces computation time when multiple processes are used.
- Increasing the number of processes does not guarantee improved end-to-end performance.
- MPI startup and communication overhead are particularly significant for smaller matrices.
- OpenMP achieves lower end-to-end execution times for most tested configurations.
- MPI approaches OpenMP performance for larger matrices.
- MPI introduces substantially higher aggregate process memory consumption due to separate address spaces and matrix replication.

Detailed results, statistical summaries and figures are available in the corresponding repository directories.

## 12. Generated Figures

| Figure | Description |
|---|---|
| runtime | End-to-end execution time comparison. |
| operation | Multiplication operation time. |
| compute | Computation time comparison. |
| mpi_scaling | MPI strong scaling. |
| mpi_speedup | MPI speedup relative to one process. |
| mpi_efficiency | MPI efficiency. |
| speedup_comparison | OpenMP and MPI speedup relative to sequential execution. |
| mpi_memory | Sum of individual MPI process peak RSS values. |
| mpi_memory_per_process | Maximum individual MPI process peak RSS. |
| memory_comparison | Memory comparison between implementations. |
| mpi_phases | MPI collective operation timings. |
| mpi_overhead | Comparison of MPI timing boundaries. |

All figures are available in PDF and PNG formats.

## 13. Limitations

- All MPI processes execute on a single physical machine.
- Inter-machine network performance is not evaluated.
- Distributed memory capacity across multiple physical nodes is not evaluated.
- Communication timings include local transfer and potential synchronisation overhead.
- Actual network transfer volumes and hardware performance counters are not measured.
- Compiler optimisation differences may affect comparisons between sequential and OpenMP kernels.
- CPU affinity is not explicitly controlled by the benchmark script.
- Five repetitions provide a limited sample of runtime variability.
- Matrix sizes above 3000 were not tested.
- Correctness validation is based on reference checksum comparisons rather than complete element-wise comparisons of every implementation.

## 14. References

- MPI Forum. MPI Standard. https://www.mpi-forum.org/docs/
- Open MPI Documentation. https://docs.open-mpi.org/
- OpenMP Architecture Review Board. https://www.openmp.org/
- Python Documentation. https://docs.python.org/3/
- Matplotlib Documentation. https://matplotlib.org/stable/

## 15. Repository

https://github.com/mariabaezz/Assignment-4-BD

This project is part of the individual assignments for the Big Data course at Universidad de Las Palmas de Gran Canaria.