import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "results_final.csv"
OUTPUT = ROOT / "results" / "summary_final.csv"
FIGURES = ROOT / "figures"

FIGURES.mkdir(exist_ok=True)

METHODS = ["seq", "omp", "mpi"]

LABELS = {
    "seq": "Sequential",
    "omp": "OpenMP",
    "mpi": "MPI"
}

COLORS = {
    "seq": "#555555",
    "omp": "#2878B5",
    "mpi": "#E07A35"
}

WORKERS = [1, 2, 4, 8, 10]

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300
})



groups = defaultdict(list)

with INPUT.open(newline="") as f:
    for row in csv.DictReader(f):

        if row["status"] != "ok":
            print("Excluded unsuccessful run:", row)
            continue

        key = (
            int(row["n"]),
            row["method"],
            int(row["workers"])
        )

        groups[key].append(row)

SIZES = sorted({key[0] for key in groups})

METRICS = [
    "compute_s",
    "operation_s",
    "end_to_end_s",
    "rss_max_bytes",
    "rss_sum_bytes",
    "setup_s",
    "bcast_s",
    "scatter_s",
    "gather_s",
    "comm_s"
]



def iqr(values):
    if len(values) < 2:
        return 0.0

    q = statistics.quantiles(
        values, n=4, method="inclusive"
    )

    return q[2] - q[0]


summary = []

for (n, method, workers), rows in sorted(groups.items()):

    result = {
        "n": n,
        "method": method,
        "workers": workers,
        "repetitions": len(rows)
    }

    for metric in METRICS:

        values = [
            float(row[metric])
            for row in rows
            if row.get(metric, "") != ""
        ]

        if values:
            result[f"median_{metric}"] = (
                statistics.median(values)
            )
            result[f"iqr_{metric}"] = iqr(values)
        else:
            result[f"median_{metric}"] = None
            result[f"iqr_{metric}"] = None

    summary.append(result)


lookup = {
    (r["n"], r["method"], r["workers"]): r
    for r in summary
}


for r in summary:

    n = r["n"]
    method = r["method"]
    p = r["workers"]

    baseline_seq = lookup[(n, "seq", 1)]
    baseline_own = lookup[(n, method, 1)]

    for metric in [
        "compute_s",
        "operation_s",
        "end_to_end_s"
    ]:

        field = f"median_{metric}"

        speedup_seq = (
            baseline_seq[field] / r[field]
        )

        speedup_own = (
            baseline_own[field] / r[field]
        )

        r[f"speedup_seq_{metric}"] = speedup_seq
        r[f"efficiency_seq_{metric}"] = speedup_seq / p

        r[f"speedup_own_{metric}"] = speedup_own
        r[f"efficiency_own_{metric}"] = speedup_own / p


# Verify checksums across methods and workers

for n in SIZES:

    checksums = {
        row["checksum"]
        for key, rows in groups.items()
        if key[0] == n
        for row in rows
    }

    if len(checksums) != 1:
        raise RuntimeError(
            f"Checksum mismatch for n={n}"
        )

    print(f"Checksum validation passed: n={n}")


# Save statistical summary

with OUTPUT.open("w", newline="") as f:

    writer = csv.DictWriter(
        f, fieldnames=list(summary[0].keys())
    )

    writer.writeheader()
    writer.writerows(summary)


def get(n, method, workers, metric):
    return lookup[(n, method, workers)][metric]


def save_plot(filename):

    plt.tight_layout()

    plt.savefig(
        FIGURES / f"{filename}.pdf",
        bbox_inches="tight"
    )

    plt.savefig(
        FIGURES / f"{filename}.png",
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")


def runtime_plot(metric, ylabel, title, filename):

    plt.figure()

    configurations = [
        ("seq", 1, "Sequential", "-"),
        ("omp", 8, "OpenMP (8 threads)", "-"),
        ("omp", 10, "OpenMP (10 threads)", "--"),
        ("mpi", 8, "MPI (8 processes)", "-"),
        ("mpi", 10, "MPI (10 processes)", "--")
    ]

    for method, p, label, style in configurations:

        times = [
            get(n, method, p, f"median_{metric}")
            for n in SIZES
        ]

        # IQR shown as a Q1-Q3 interval.
        lower = []
        upper = []

        for n in SIZES:

            values = [
                float(r[metric])
                for r in groups[(n, method, p)]
            ]

            q = statistics.quantiles(
                values, n=4, method="inclusive"
            )

            lower.append(q[0])
            upper.append(q[2])

        plt.plot(
            SIZES,
            times,
            marker="o",
            linestyle=style,
            color=COLORS[method],
            label=label
        )

        plt.fill_between(
            SIZES,
            lower,
            upper,
            color=COLORS[method],
            alpha=0.10
        )

    plt.xlabel("Matrix size (n)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(SIZES)
    plt.legend()

    save_plot(filename)


runtime_plot(
    "end_to_end_s",
    "Execution time (s)",
    "End-to-end execution time",
    "runtime"
)

runtime_plot(
    "operation_s",
    "Operation time (s)",
    "Matrix multiplication operation time",
    "operation"
)

runtime_plot(
    "compute_s",
    "Computation time (s)",
    "Computation time comparison",
    "compute"
)


plt.figure()

for n in SIZES:

    times = [
        get(n, "mpi", p, "median_end_to_end_s")
        for p in WORKERS
    ]

    plt.plot(
        WORKERS,
        times,
        marker="o",
        label=f"n = {n}"
    )

plt.xlabel("MPI processes")
plt.ylabel("End-to-end time (s)")
plt.title("MPI strong scaling")
plt.xticks(WORKERS)
plt.legend()

save_plot("mpi_scaling")



plt.figure()

plt.plot(
    WORKERS,
    WORKERS,
    "--",
    color="gray",
    label="Ideal speedup"
)

for n in SIZES:

    speedup = [
        get(n, "mpi", p, "speedup_own_end_to_end_s")
        for p in WORKERS
    ]

    plt.plot(
        WORKERS,
        speedup,
        marker="o",
        label=f"n = {n}"
    )

plt.xlabel("MPI processes")
plt.ylabel("Speedup relative to MPI with one process")
plt.title("MPI end-to-end speedup")
plt.xticks(WORKERS)
plt.legend()

save_plot("mpi_speedup")



plt.figure()

plt.axhline(
    100,
    linestyle="--",
    color="gray",
    label="Ideal efficiency"
)

for n in SIZES:

    efficiency = [
        100 * get(
            n, "mpi", p,
            "efficiency_own_end_to_end_s"
        )
        for p in WORKERS
    ]

    plt.plot(
        WORKERS,
        efficiency,
        marker="o",
        label=f"n = {n}"
    )

plt.xlabel("MPI processes")
plt.ylabel("Efficiency (%)")
plt.title("MPI end-to-end efficiency")
plt.xticks(WORKERS)
plt.legend()

save_plot("mpi_efficiency")



n = max(SIZES)

plt.figure()

plt.plot(
    WORKERS,
    WORKERS,
    "--",
    color="gray",
    label="Ideal speedup"
)

for method in ["omp", "mpi"]:

    speedup = [
        get(
            n, method, p,
            "speedup_seq_end_to_end_s"
        )
        for p in WORKERS
    ]

    plt.plot(
        WORKERS,
        speedup,
        marker="o",
        color=COLORS[method],
        label=LABELS[method]
    )

plt.xlabel("Threads / MPI processes")
plt.ylabel("Speedup relative to sequential")
plt.title(f"End-to-end speedup comparison (n = {n})")
plt.xticks(WORKERS)
plt.legend()

save_plot("speedup_comparison")


plt.figure()

for n in SIZES:

    memory = [
        get(n, "mpi", p, "median_rss_sum_bytes")
        / 1e6
        for p in WORKERS
    ]

    plt.plot(
        WORKERS,
        memory,
        marker="o",
        label=f"n = {n}"
    )

plt.xlabel("MPI processes")
plt.ylabel("Sum of individual peak RSS (MB)")
plt.title("MPI aggregate peak RSS")
plt.xticks(WORKERS)
plt.legend()

save_plot("mpi_memory")


plt.figure()

for n in SIZES:

    memory = [
        get(n, "mpi", p, "median_rss_max_bytes")
        / 1e6
        for p in WORKERS
    ]

    plt.plot(
        WORKERS,
        memory,
        marker="o",
        label=f"n = {n}"
    )

plt.xlabel("MPI processes")
plt.ylabel("Maximum individual peak RSS (MB)")
plt.title("Maximum memory per MPI process")
plt.xticks(WORKERS)
plt.legend()

save_plot("mpi_memory_per_process")


plt.figure()

configurations = [
    ("seq", 1, "Sequential"),
    ("omp", 10, "OpenMP (10 threads)"),
    ("mpi", 8, "MPI (8 processes)"),
    ("mpi", 10, "MPI (10 processes)")
]

for method, p, label in configurations:

    memory = [
        get(n, method, p, "median_rss_sum_bytes")
        / 1e6
        for n in SIZES
    ]

    plt.plot(
        SIZES,
        memory,
        marker="o",
        color=COLORS[method],
        linestyle="--" if method == "mpi" and p == 10 else "-",
        label=label
    )

plt.xlabel("Matrix size (n)")
plt.ylabel("Sum of process peak RSS (MB)")
plt.title("Memory consumption comparison")
plt.xticks(SIZES)
plt.legend()

save_plot("memory_comparison")



n = max(SIZES)

plt.figure()

phases = [
    ("bcast_s", "Broadcast"),
    ("scatter_s", "Scatter"),
    ("gather_s", "Gather")
]

for metric, label in phases:

    times = [
        get(n, "mpi", p, f"median_{metric}")
        for p in WORKERS
    ]

    plt.plot(
        WORKERS,
        times,
        marker="o",
        label=label
    )

plt.xlabel("MPI processes")
plt.ylabel("Phase duration (s)")
plt.title(f"MPI collective operation timings (n = {n})")
plt.xticks(WORKERS)
plt.legend()

save_plot("mpi_phases")



plt.figure()

metrics = [
    ("compute_s", "Computation"),
    ("operation_s", "Distributed operation"),
    ("end_to_end_s", "End-to-end")
]

for metric, label in metrics:

    times = [
        get(n, "mpi", p, f"median_{metric}")
        for p in WORKERS
    ]

    plt.plot(
        WORKERS,
        times,
        marker="o",
        label=label
    )

plt.xlabel("MPI processes")
plt.ylabel("Time (s)")
plt.title(f"MPI timing boundaries (n = {n})")
plt.xticks(WORKERS)
plt.legend()

save_plot("mpi_overhead")



print("\nAnalysis completed successfully.")
print(f"Configurations: {len(summary)}")
print(f"Measurements: {sum(len(r) for r in groups.values())}")
print(f"Summary: {OUTPUT}")
print(f"Figures: {FIGURES}")