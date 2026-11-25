import subprocess
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SIZES = [1, 2, 3, 7, 11, 32, 100]
WORKERS = [1, 2, 4, 8, 10]


def generate_matrix(n, seed):
    x = seed
    values = []

    for _ in range(n * n):
        x = (1664525 * x + 1013904223) & 0xFFFFFFFF
        values.append(x % 10)

    return [
        values[i * n:(i + 1) * n]
        for i in range(n)
    ]


def multiply_reference(A, B):
    n = len(A)

    return [
        [
            sum(A[i][k] * B[k][j] for k in range(n))
            for j in range(n)
        ]
        for i in range(n)
    ]


def reference_checksum(n):
    A = generate_matrix(n, 42)
    B = generate_matrix(n, 4242)

    C = multiply_reference(A, B)

    return sum(sum(row) for row in C)


def execute(command):
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120
    )

    match = re.search(
        r"checksum=(\d+)",
        result.stdout
    )

    if not match:
        raise RuntimeError(result.stdout)

    return int(match.group(1))


def test_correctness():
    passed = 0

    for n in SIZES:
        expected = reference_checksum(n)

        configurations = [
            ("seq", 1)
        ] + [
            (method, p)
            for method in ("omp", "mpi")
            for p in WORKERS
        ]

        for method, p in configurations:

            if method == "mpi":
                command = [
                    "mpirun", "-np", str(p),
                    "./bin/mpi_matmul", str(n)
                ]
            else:
                command = [
                    "./bin/run_parallel",
                    method, str(n), str(p)
                ]

            actual = execute(command)

            assert actual == expected, (
                f"FAILED: n={n}, method={method}, "
                f"workers={p}, expected={expected}, "
                f"actual={actual}"
            )

            print(
                f"PASS: n={n}, method={method}, "
                f"workers={p}, checksum={actual}"
            )

            passed += 1

    print(f"\nAll {passed} correctness tests passed.")


if __name__ == "__main__":
    test_correctness()