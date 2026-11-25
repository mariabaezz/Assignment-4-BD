import csv
import os
import random
import re
import statistics
import subprocess
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"

TIMEOUT = 120
MEMORY_BUDGET = 4 * 1024**3
REPETITIONS = 5

rng = random.Random(42)

FIELDS = [
    "n",
    "method",
    "workers",
    "repetition",
    "diagnostics",
    "compute_s",
    "operation_s",
    "end_to_end_s",
    "setup_s",
    "bcast_s",
    "scatter_s",
    "gather_s",
    "compute_min_s",
    "compute_mean_s",
    "compute_max_s",
    "wait_max_s",
    "gather_sync_s",
    "imbalance_ratio",
    "rss_max_bytes",
    "rss_sum_bytes",
    "checksum",
    "status"
]


def run(n, method, workers, rep, diagnostics=False):

    row = {
        "n": n,
        "method": method,
        "workers": workers,
        "repetition": rep,
        "diagnostics": int(diagnostics)
    }

    # Matrix-only memory estimate.
    estimated_memory = (
        8 * n**2 * (workers + 4)
        if method == "mpi"
        else 24 * n**2
    )

    if estimated_memory > MEMORY_BUDGET:
        row["status"] = "estimated_memory_limit"
        return row

    env = os.environ.copy()

    env.pop("MPI_DIAGNOSTICS", None)

    if diagnostics:
        env["MPI_DIAGNOSTICS"] = "1"

    if method == "mpi":
        command = [
            "mpirun", "-np", str(workers),
            "./bin/mpi_matmul", str(n)
        ]
    else:
        command = [
            "./bin/run_parallel",
            method, str(n), str(workers)
        ]

    start = time.perf_counter()

    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=True
        )

    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
        return row

    except subprocess.CalledProcessError as error:
        print(error.stderr)
        row["status"] = "error"
        return row

    elapsed = time.perf_counter() - start

    data = dict(
        re.findall(
            r"(\w+)=([^\s,]+)",
            result.stdout
        )
    )

    row.update({
        "compute_s": float(
            data["compute"]
            if method == "mpi"
            else data["time"]
        ),
        "operation_s": float(
            data["total"]
            if method == "mpi"
            else data["time"]
        ),
        "end_to_end_s": elapsed,
        "rss_max_bytes": int(data["rss_max_bytes"]),
        "rss_sum_bytes": int(
            data.get(
                "rss_sum_bytes",
                data["rss_max_bytes"]
            )
        ),
        "checksum": int(data["checksum"]),
        "status": "ok"
    })

    for key in [
        "setup",
        "bcast",
        "scatter",
        "gather",
        "compute_min",
        "compute_mean",
        "compute_max",
        "wait_max",
        "gather_sync"
    ]:
        if key in data:
            row[key + "_s"] = float(data[key])

    if "imbalance_ratio" in data:
        row["imbalance_ratio"] = float(
            data["imbalance_ratio"]
        )

    if row["rss_sum_bytes"] > MEMORY_BUDGET:
        row["status"] = "memory_budget_exceeded"

    return row


def experiment(filename, configs):

    checksums = {}
    failed = set()
    rows = []

    with (RESULTS / filename).open(
        "w", newline=""
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=FIELDS
        )

        writer.writeheader()

        for rep in range(REPETITIONS + 1):

            order = configs.copy()
            rng.shuffle(order)

            for config in order:

                if config in failed:
                    continue

                n, method, workers, diag = config

                row = run(
                    n, method, workers, rep, diag
                )

                if "checksum" in row:

                    expected = checksums.setdefault(
                        n, row["checksum"]
                    )

                    if row["checksum"] != expected:
                        row["status"] = "checksum_mismatch"

                if row["status"] != "ok":
                    failed.add(config)

                # Exclude successful warm-up executions.
                # Preserve failures even during warm-up.

                if rep > 0 or row["status"] != "ok":
                    writer.writerow(row)
                    file.flush()
                    rows.append(row)

                print(
                    f"n={n} {method} p={workers} "
                    f"diag={diag} rep={rep} "
                    f"status={row['status']} "
                    f"time={row.get('end_to_end_s', '-')}",
                    flush=True
                )

    return rows


def summarize(rows):

    keys = sorted({
        (
            r["n"],
            r["method"],
            r["workers"],
            r["diagnostics"]
        )
        for r in rows
    })

    for key in keys:

        selected = [
            r for r in rows
            if (
                r["n"],
                r["method"],
                r["workers"],
                r["diagnostics"]
            ) == key
            and r["status"] == "ok"
        ]

        if not selected:
            continue

        print("\nConfiguration:", key)
        print("Measurements:", len(selected))

        metrics = [
            "compute_s",
            "operation_s",
            "end_to_end_s",
            "gather_s",
            "compute_min_s",
            "compute_mean_s",
            "compute_max_s",
            "wait_max_s",
            "gather_sync_s",
            "imbalance_ratio",
            "rss_sum_bytes"
        ]

        for metric in metrics:

            values = [
                r[metric] for r in selected
                if metric in r
            ]

            if values:

                median = statistics.median(values)

                if metric == "rss_sum_bytes":
                    print(
                        f"  {metric}: "
                        f"{median / 1e6:.2f} MB"
                    )
                else:
                    print(
                        f"  {metric}: {median:.6f}"
                    )


def main():

    RESULTS.mkdir(exist_ok=True)

    print("Runtime budget:", TIMEOUT, "seconds")
    print("Memory budget: 4 GiB")
    print("Maximum matrix dimension: 4000")

    # Experiment 1:
    # Compare ordinary and synchronized MPI gather.

    diagnostics_configs = [
        (3000, "mpi", p, diag)
        for p in [1, 2, 4, 8, 10]
        for diag in [False, True]
    ]

    diagnostics = experiment(
        "results_diagnostics.csv",
        diagnostics_configs
    )

    # Experiment 2:
    # Larger workload with a declared resource budget.

    extension_configs = [
        (4000, "seq", 1, False),
        (4000, "omp", 1, False),
        (4000, "omp", 10, False),
        (4000, "mpi", 1, False),
        (4000, "mpi", 8, False),
        (4000, "mpi", 10, False)
    ]

    extension = experiment(
        "results_extension.csv",
        extension_configs
    )

    print("\n=== MPI DIAGNOSTICS ===")
    summarize(diagnostics)

    print("\n=== SIZE EXTENSION ===")
    summarize(extension)


if __name__ == "__main__":
    main()