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
│   ├── benchmark_diagnostics.py
│   └── plots.py
├── tests/
│   ├── test_correctness.py
│   ├── test_elementwise.py
│   └── verify_checksum.py
├── results/
│   ├── results_final.csv
│   ├── summary_final.csv
│   ├── correctness_tests.txt
│   ├── results_diagnostics.csv
│   ├── results_extension.csv
│   └── diagnostics_summary.txt
├── figures/
├── preliminary/
├── report/
│   ├── main.tex
│   └── Assignment04_BaezMarrero_Report.pdf
├── environment.txt
├── requirements.txt
├── Makefile
├── .gitignore
└── README.md
```

The `bin/` directory contains locally compiled executables
and is excluded from version control.

The independent checksum verification for n = 4000
can be reproduced using:

```bash
python3 tests/verify_checksum.py 4000
```

Expected output:

```text
n=4000,checksum=1296165693892
```

This verification uses an O(n²) identity based on column
sums of A and row sums of B, without computing the complete
matrix product.

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

Correctness was evaluated using two independent validation suites.

```bash
python3 tests/test_correctness.py
python3 tests/test_elementwise.py
```

### Checksum validation

The first suite performs 77 checksum comparisons against an independent Python reference implementation.

It covers seven matrix sizes and eleven sequential, OpenMP and MPI configurations.

### Element-wise validation

The second suite performs 121 exact element-wise comparisons across eleven test cases and eleven execution configurations.

The test cases include:

- Multiplication by an identity matrix on both sides.
- Multiplication by a zero matrix on both sides.
- A known 2 × 2 matrix multiplication.
- Deterministically generated matrices.
- Uneven row partitioning.
- Configurations with more MPI processes than matrix rows.

The independent reference uses a conventional i-j-k implementation in Python.

All 121 element-wise tests passed successfully. The validation output is available in `results/correctness_tests.txt`.

Additionally, all 275 benchmark measurements produced matching checksums across implementations for each matrix size.

The validation mode was added after the performance measurements without changing the multiplication kernels or normal input generation. Subsequent executions confirmed that the expected benchmark checksum remained unchanged.

### Numerical safety

All input elements are integers between 0 and 9. For n = 3000, the theoretical maximum value of an output element is 243,000, while the maximum possible checksum is 2,187,000,000,000.

Both values are within the representable range of 64-bit integers. Therefore, integer overflow is not expected under the tested experimental conditions, and exact equality is used instead of floating-point tolerances.

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

### Resource limits

Each execution was subject to a timeout of 120 seconds.

The largest attempted and successfully completed matrix size was n = 3000. All 275 recorded benchmark executions completed successfully.

The machine provides 16 GiB of RAM. The largest observed sum of individual process peak RSS values was approximately 1.15 GB.

An explicit memory cutoff was not established before the experiments, and the maximum feasible matrix size of the hardware was not investigated. Therefore, n = 3000 represents the largest tested size rather than the maximum hardware capacity.

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

- All experiments were conducted on a single physical Apple M4 machine.
- MPI results characterise local multi-process execution rather than inter-machine network performance.
- Distributed memory capacity across multiple physical nodes was not investigated.
- Communication measurements include potential synchronisation delays and local data movement.
- Individual MPI phase maxima are not necessarily additive.
- Actual physical data-transfer volumes were not measured.
- Aggregate process peak RSS is not equivalent to simultaneous physical memory consumption.
- Hardware performance counters were not collected.
- Compiler optimisation differences may affect comparisons between sequential and OpenMP kernels.
- CPU affinity was not explicitly controlled by the benchmark script.
- The heterogeneous CPU architecture may influence scalability.
- Five repetitions provide a limited sample of runtime variability.
- No explicit memory cutoff was established before benchmarking.
- Matrix sizes above 3000 were not tested.
- Full element-wise validation was performed on small matrices, while larger benchmark configurations were validated through checksum agreement.

## 14. Supplementary Experiments

Two supplementary experiments were conducted to investigate MPI computation imbalance, synchronization costs and performance at a larger matrix dimension.

Before these experiments, the following resource budgets were established:

- Maximum runtime: 120 seconds per execution.
- Memory acceptance budget: 4 GiB, evaluated using summed process peak RSS.
- Maximum matrix dimension: 4000.
- One warm-up and five measured repetitions per configuration.

The memory budget is an experimental acceptance criterion, not an operating-system-enforced memory limit.

### MPI diagnostics

The optional diagnostic mode can be enabled using:

```bash
MPI_DIAGNOSTICS=1 mpirun -np 8 ./bin/mpi_matmul 3000
```

The diagnostic mode introduces a barrier before result collection and reports minimum, mean and maximum computation durations, maximum barrier waiting time, synchronized gather duration and the computation imbalance ratio.

The supplementary experiment compares synchronized and unsynchronized execution across five MPI process configurations.

The results indicate that the original gather measurements include waiting associated with differences in process arrival times. They also reveal that computation time is the principal scaling limitation for larger workloads.

### Size extension

An additional experiment evaluates six selected configurations at n = 4000.

All configurations completed successfully within the declared resource budgets.

MPI with eight processes achieved the lowest median end-to-end execution time, approximately 6.834 seconds.

### Reproduction

```bash
python3 scripts/benchmark_diagnostics.py
```

The supplementary results are stored in:

- `results/results_diagnostics.csv`
- `results/results_extension.csv`
- `results/diagnostics_summary.txt`

The supplementary experiments contain 80 measured executions, bringing the total to 355.

The original 275 benchmark measurements and 12 performance figures remain unchanged.

The MPI implementation accepts matrix dimensions between 1 and 4000. The Makefile targets Apple Silicon macOS with Homebrew-installed dependencies.

Full element-wise correctness validation is available through `tests/test_elementwise.py`.

## 15. References

- MPI Forum. MPI Standard. https://www.mpi-forum.org/docs/
- Open MPI Documentation. https://docs.open-mpi.org/
- OpenMP Architecture Review Board. https://www.openmp.org/
- Python Documentation. https://docs.python.org/3/
- Matplotlib Documentation. https://matplotlib.org/stable/

## 16. Repository

https://github.com/mariabaezz/Assignment-4-BD

This project is part of the individual assignments for the Big Data course at Universidad de Las Palmas de Gran Canaria.

## Additional Reproducibility Materials

- `tests/test_elementwise.py`: exact element-wise validation.
- `tests/verify_checksum.py`: independent O(n²) checksum verification.
- `scripts/benchmark_diagnostics.py`: supplementary experiments.
- `results/results_diagnostics.csv`: MPI synchronization measurements.
- `results/results_extension.csv`: n=4000 measurements.
- `results/diagnostics_summary.txt`: supplementary statistical summary.
- `report/main.tex`: LaTeX report source.
- `report/Assignment04_BaezMarrero_Report.pdf`: final report.