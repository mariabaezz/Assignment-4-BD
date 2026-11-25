#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "Allocation failed for %zu bytes\n", bytes);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return p;
}

static void fill_matrix(int64_t *M, int n, uint64_t seed) {
    uint64_t x = seed ? seed : 1;
    size_t total = (size_t)n * (size_t)n;
    for (size_t i = 0; i < total; ++i) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        M[i] = (int64_t)(x % 10ULL);  // values 0..9
    }
}

static uint64_t checksum(const int64_t *C, size_t count) {
    uint64_t s = 0;
    for (size_t i = 0; i < count; ++i) s += (uint64_t)C[i];
    return s;
}

static void multiply_rows_ikj(const int64_t *A_local, const int64_t *B,
                              int64_t *C_local, int local_rows, int n) {
    memset(C_local, 0, (size_t)local_rows * (size_t)n * sizeof(int64_t));
    for (int i = 0; i < local_rows; ++i) {
        for (int k = 0; k < n; ++k) {
            const int64_t aik = A_local[(size_t)i * n + k];
            for (int j = 0; j < n; ++j) {
                C_local[(size_t)i * n + j] += aik * B[(size_t)k * n + j];
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    if (argc < 2 || argc > 3) {
        if (rank == 0) fprintf(stderr, "Usage: %s <n> [seed]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    uint64_t seed = (argc == 3) ? strtoull(argv[2], NULL, 10) : 12345ULL;
    if (n <= 0) {
        if (rank == 0) fprintf(stderr, "n must be > 0\n");
        MPI_Finalize();
        return 1;
    }

    int base = n / world;
    int rem = n % world;
    int local_rows = base + (rank < rem ? 1 : 0);

    int *sendcounts = xmalloc((size_t)world * sizeof(int));
    int *displs = xmalloc((size_t)world * sizeof(int));
    int offset = 0;
    for (int r = 0; r < world; ++r) {
        int rows_r = base + (r < rem ? 1 : 0);
        sendcounts[r] = rows_r * n;
        displs[r] = offset;
        offset += sendcounts[r];
    }

    int64_t *A = NULL;
    int64_t *C = NULL;
    int64_t *B = xmalloc((size_t)n * (size_t)n * sizeof(int64_t));
    int64_t *A_local = xmalloc((size_t)local_rows * (size_t)n * sizeof(int64_t));
    int64_t *C_local = xmalloc((size_t)local_rows * (size_t)n * sizeof(int64_t));

    double setup_start = MPI_Wtime();
    if (rank == 0) {
        A = xmalloc((size_t)n * (size_t)n * sizeof(int64_t));
        C = xmalloc((size_t)n * (size_t)n * sizeof(int64_t));
        fill_matrix(A, n, seed);
        fill_matrix(B, n, seed + 1);
    }
    double setup_local = MPI_Wtime() - setup_start;

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start = MPI_Wtime();

    double bcast_start = MPI_Wtime();
    MPI_Bcast(B, n * n, MPI_INT64_T, 0, MPI_COMM_WORLD);
    double bcast_local = MPI_Wtime() - bcast_start;

    double scatter_start = MPI_Wtime();
    MPI_Scatterv(A, sendcounts, displs, MPI_INT64_T,
                 A_local, local_rows * n, MPI_INT64_T,
                 0, MPI_COMM_WORLD);
    double scatter_local = MPI_Wtime() - scatter_start;

    MPI_Barrier(MPI_COMM_WORLD);
    double compute_start = MPI_Wtime();
    multiply_rows_ikj(A_local, B, C_local, local_rows, n);
    double compute_local = MPI_Wtime() - compute_start;

    double gather_start = MPI_Wtime();
    MPI_Gatherv(C_local, local_rows * n, MPI_INT64_T,
                C, sendcounts, displs, MPI_INT64_T,
                0, MPI_COMM_WORLD);
    double gather_local = MPI_Wtime() - gather_start;

    double total_local = MPI_Wtime() - total_start;

    double setup_max, bcast_max, scatter_max, compute_max, gather_max, total_max;
    MPI_Reduce(&setup_local, &setup_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bcast_local, &bcast_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&scatter_local, &scatter_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_local, &compute_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&gather_local, &gather_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_local, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        uint64_t sum = checksum(C, (size_t)n * (size_t)n);
        double comm = bcast_max + scatter_max + gather_max;
        printf("n=%d workers=%d seed=%" PRIu64
               " setup=%.9f bcast=%.9f scatter=%.9f compute=%.9f gather=%.9f comm=%.9f total=%.9f checksum=%" PRIu64 "\n",
               n, world, seed, setup_max, bcast_max, scatter_max,
               compute_max, gather_max, comm, total_max, sum);
    }

    free(A);
    free(B);
    free(C);
    free(A_local);
    free(C_local);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
