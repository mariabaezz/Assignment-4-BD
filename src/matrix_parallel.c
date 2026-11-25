#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <sys/resource.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static long long *alloc_matrix(int n) {
    return (long long *)calloc((size_t)n * (size_t)n, sizeof(long long));
}

static void fill_matrix(long long *M, int n, unsigned int seed) {
    // Deterministic generator. Values 0..9.
    uint32_t x = seed;
    for (size_t i = 0; i < (size_t)n * (size_t)n; ++i) {
        x = 1664525u * x + 1013904223u;
        M[i] = (long long)(x % 10u);
    }
}

static void multiply_reordered_seq(
    const long long *A,
    const long long *B,
    long long *C,
    int n
) {
    memset(C, 0, (size_t)n * (size_t)n * sizeof(long long));

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            long long aik = A[(size_t)i * n + k];
            for (int j = 0; j < n; ++j) {
                C[(size_t)i * n + j] += aik * B[(size_t)k * n + j];
            }
        }
    }
}

static void multiply_reordered_omp(
    const long long *A,
    const long long *B,
    long long *C,
    int n,
    int threads
) {
    memset(C, 0, (size_t)n * (size_t)n * sizeof(long long));

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    // Each iteration owns one complete row C[i,*], so different threads
    // never write the same output elements. A and B are read-only.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            long long aik = A[(size_t)i * n + k];
            for (int j = 0; j < n; ++j) {
                C[(size_t)i * n + j] += aik * B[(size_t)k * n + j];
            }
        }
    }
}

static unsigned long long checksum(const long long *C, int n) {
    unsigned long long s = 0;
    for (size_t i = 0; i < (size_t)n * (size_t)n; ++i) {
        s += (unsigned long long)C[i];
    }
    return s;
}

static int matrices_equal(const long long *X, const long long *Y, int n) {
    for (size_t i = 0; i < (size_t)n * (size_t)n; ++i) {
        if (X[i] != Y[i]) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s <seq|omp> <n> <threads>\n", argv[0]);
        return 1;
    }

    const char *method = argv[1];
    int n = atoi(argv[2]);
    int threads = atoi(argv[3]);

    if (n <= 0 || threads <= 0) {
        fprintf(stderr, "n and threads must be positive integers.\n");
        return 1;
    }

    long long *A = alloc_matrix(n);
    long long *B = alloc_matrix(n);
    long long *C = alloc_matrix(n);

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed for n=%d.\n", n);
        free(A); free(B); free(C);
        return 2;
    }

    fill_matrix(A, n, 42u);
    fill_matrix(B, n, 4242u);

    const char *test = argc == 5 ? argv[4] : NULL;

    if (test) {
        if (n > 16) return 4;

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
            return 4;
        }
    }

    double t0 = now_seconds();

    if (strcmp(method, "seq") == 0) {
        multiply_reordered_seq(A, B, C, n);
    } else if (strcmp(method, "omp") == 0) {
        multiply_reordered_omp(A, B, C, n, threads);
    } else {
        fprintf(stderr, "Unknown method: %s\n", method);
        free(A); free(B); free(C);
        return 1;
    }

    double t1 = now_seconds();

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    unsigned long long rss = (unsigned long long)ru.ru_maxrss;
#ifndef __APPLE__
    rss *= 1024;
#endif
    if (n <= 16) {
        for (int i=0;i<n;i++) for(int j=0;j<n;j++) {
            long long expected=0;
            for(int k=0;k<n;k++) expected += A[(size_t)i*n+k]*B[(size_t)k*n+j];
            if(C[(size_t)i*n+j]!=expected) return 3;
        }
    }
    printf("method=%s,n=%d,threads=%d,time=%.9f,checksum=%llu,rss_max_bytes=%llu\n",
           method, n, threads, t1 - t0, checksum(C, n), rss);

    if (test) {
        printf("RESULT=");
        for (size_t i=0; i<(size_t)n*n; i++)
            printf("%s%lld", i ? "," : "", C[i]);
        printf("\n");
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
