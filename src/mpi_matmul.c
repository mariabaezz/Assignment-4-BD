#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/resource.h>

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p && bytes) {
        fprintf(stderr, "Allocation failed for %zu bytes\n", bytes);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return p;
}

static void fill_matrix(int64_t *M, int n, uint64_t seed) {
    uint32_t x = (uint32_t)seed;
    for (size_t i = 0; i < (size_t)n*n; ++i) {
        x = 1664525u * x + 1013904223u;
        M[i] = x % 10u;
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

    if (argc < 2 || argc > 4) {
        if (rank == 0) fprintf(stderr, "Usage: %s <n> [seed]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    uint64_t seed = (argc >= 3) ? strtoull(argv[2], NULL, 10) : 42ULL;
    if (n <= 0 || n > 4000) {
        if (rank == 0) fprintf(stderr, "n must be > 0\n");
        MPI_Finalize();
        return 1;
    }

    const char *test = argc == 4 ? argv[3] : NULL;

    if (test && n > 16)
        MPI_Abort(MPI_COMM_WORLD, 4);

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
        fill_matrix(B, n, 4242u);

        if (test) {
            if (strcmp(test, "right_identity") == 0) {
                memset(B, 0, (size_t)n*n*sizeof(*B));
                for (int i=0; i<n; i++) B[i*n+i] = 1;
            } else if (strcmp(test, "left_identity") == 0) {
                memset(A, 0, (size_t)n*n*sizeof(*A));
                for (int i=0; i<n; i++) A[i*n+i] = 1;
            } else if (strcmp(test, "right_zero") == 0) {
                memset(B, 0, (size_t)n*n*sizeof(*B));
            } else if (strcmp(test, "left_zero") == 0) {
                memset(A, 0, (size_t)n*n*sizeof(*A));
            } else if (strcmp(test, "known") == 0 && n == 2) {
                A[0]=1; A[1]=2; A[2]=3; A[3]=4;
                B[0]=5; B[1]=6; B[2]=7; B[3]=8;
            } else if (strcmp(test, "random") != 0) {
                MPI_Abort(MPI_COMM_WORLD, 4);
            }
        }
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

    MPI_Barrier(MPI_COMM_WORLD);
    double total_local = MPI_Wtime() - total_start;

    double setup_max, bcast_max, scatter_max, compute_max, gather_max, total_max;
    MPI_Reduce(&setup_local, &setup_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bcast_local, &bcast_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&scatter_local, &scatter_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_local, &compute_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&gather_local, &gather_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_local, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    unsigned long long rss = (unsigned long long)ru.ru_maxrss;
#ifndef __APPLE__
    rss *= 1024;
#endif
    unsigned long long rss_max, rss_sum;
    MPI_Reduce(&rss, &rss_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rss, &rss_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    int valid = 1;
    if (rank == 0 && n <= 16) {
        for (int i=0;i<n;i++) for(int j=0;j<n;j++) {
            int64_t expected=0;
            for(int k=0;k<n;k++) expected += A[(size_t)i*n+k]*B[(size_t)k*n+j];
            if(C[(size_t)i*n+j]!=expected) valid=0;
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!valid) MPI_Abort(MPI_COMM_WORLD, 3);
    if (rank == 0) {
        uint64_t sum = checksum(C, (size_t)n * (size_t)n);
        double comm = bcast_max + scatter_max + gather_max;
        printf("n=%d workers=%d seed=%" PRIu64
               " setup=%.9f bcast=%.9f scatter=%.9f compute=%.9f gather=%.9f comm=%.9f total=%.9f checksum=%" PRIu64 " rss_max_bytes=%llu rss_sum_bytes=%llu\n",
               n, world, seed, setup_max, bcast_max, scatter_max,
               compute_max, gather_max, comm, total_max, sum, rss_max, rss_sum);
    }

    if (rank == 0 && test) {
        printf("RESULT=");
        for (size_t i=0; i<(size_t)n*n; i++)
            printf("%s%" PRId64, i ? "," : "", C[i]);
        printf("\n");
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
