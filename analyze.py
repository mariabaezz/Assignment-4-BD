import csv
import statistics
from collections import defaultdict

INPUT = "results_mpi.csv"
OUTPUT = "summary_mpi.csv"

rows = []
with open(INPUT, newline="") as f:
    for row in csv.DictReader(f):
        for k in ["n", "workers", "repetition"]:
            row[k] = int(row[k])
        for k in ["setup_s", "bcast_s", "scatter_s", "compute_s", "gather_s", "comm_s", "total_s"]:
            row[k] = float(row[k])
        rows.append(row)

groups = defaultdict(list)
for row in rows:
    groups[(row["n"], row["workers"])].append(row)


def iqr(values):
    values = sorted(values)
    if len(values) < 4:
        return 0.0
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[2] - q[0]

summary = []
for (n, p), rs in sorted(groups.items()):
    total = [r["total_s"] for r in rs]
    compute = [r["compute_s"] for r in rs]
    comm = [r["comm_s"] for r in rs]
    summary.append({
        "n": n,
        "workers": p,
        "median_total_s": statistics.median(total),
        "iqr_total_s": iqr(total),
        "median_compute_s": statistics.median(compute),
        "median_comm_s": statistics.median(comm),
        "repetitions": len(rs),
    })

baseline = {(r["n"]): r["median_total_s"] for r in summary if r["workers"] == 1}
for r in summary:
    t1 = baseline.get(r["n"])
    r["speedup_vs_mpi1"] = (t1 / r["median_total_s"]) if t1 else float("nan")
    r["efficiency_vs_mpi1"] = r["speedup_vs_mpi1"] / r["workers"]

fields = ["n", "workers", "median_total_s", "iqr_total_s", "median_compute_s",
          "median_comm_s", "speedup_vs_mpi1", "efficiency_vs_mpi1", "repetitions"]
with open(OUTPUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(summary)

print(f"Saved {OUTPUT}")
for r in summary:
    print(r)
