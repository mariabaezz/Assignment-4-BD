import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WORKERS = [1, 2, 4, 8, 10]

CONFIGS = [("seq", 1)] + [
    (method, p)
    for method in ("omp", "mpi")
    for p in WORKERS
]

CASES = [
    (2, "known"),
    (3, "right_identity"),
    (3, "left_identity"),
    (3, "right_zero"),
    (3, "left_zero"),
    (1, "random"),
    (2, "random"),
    (3, "random"),
    (7, "random"),
    (11, "random"),
    (16, "random")
]


def generate_matrix(n, seed):
    x = seed
    matrix = []

    for _ in range(n * n):
        x = (1664525 * x + 1013904223) & 0xFFFFFFFF
        matrix.append(x % 10)

    return matrix


def multiply_reference(A, B, n):
    # Independent conventional i-j-k implementation.

    return [
        sum(
            A[i * n + k] * B[k * n + j]
            for k in range(n)
        )
        for i in range(n)
        for j in range(n)
    ]


def expected_result(n, case):

    A = generate_matrix(n, 42)
    B = generate_matrix(n, 4242)

    if case == "known":
        return [19, 22, 43, 50]

    if case == "right_identity":
        return A

    if case == "left_identity":
        return B

    if case in ("right_zero", "left_zero"):
        return [0] * (n * n)

    if case == "random":
        return multiply_reference(A, B, n)

    raise ValueError(f"Unknown test case: {case}")


def execute(method, workers, n, case):

    if method == "mpi":
        command = [
            "mpirun",
            "-np", str(workers),
            str(ROOT / "bin" / "mpi_matmul"),
            str(n),
            "42",
            case
        ]

    else:
        command = [
            str(ROOT / "bin" / "run_parallel"),
            method,
            str(n),
            str(workers),
            case
        ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120
    )

    for line in result.stdout.splitlines():

        if line.startswith("RESULT="):

            return [
                int(value)
                for value in line.split("=", 1)[1].split(",")
            ]

    raise RuntimeError(
        f"Matrix output not found:\n{result.stdout}"
    )


def main():

    passed = 0

    for n, case in CASES:

        expected = expected_result(n, case)

        for method, workers in CONFIGS:

            actual = execute(
                method, workers, n, case
            )

            if actual != expected:

                raise AssertionError(
                    f"FAILED: {case}, n={n}, "
                    f"method={method}, workers={workers}\n"
                    f"Expected: {expected}\n"
                    f"Actual: {actual}"
                )

            print(
                f"PASS | {case:16} | "
                f"n={n:2} | {method:3} | "
                f"workers={workers:2}"
            )

            passed += 1

    print(
        f"\nAll {passed} element-wise tests passed."
    )


if __name__ == "__main__":
    main()