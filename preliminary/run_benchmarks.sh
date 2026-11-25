#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-results_mpi.csv}
REPS=${REPS:-5}
SIZES=${SIZES:-"500 1000 1500 2000"}
WORKERS=${WORKERS:-"1 2 4 8 10"}
SEED=${SEED:-12345}

if [[ ! -x ./mpi_matmul ]]; then
  echo "mpi_matmul not found. Run: make" >&2
  exit 1
fi

echo "n,workers,repetition,setup_s,bcast_s,scatter_s,compute_s,gather_s,comm_s,total_s,checksum" > "$OUT"

for n in $SIZES; do
  for p in $WORKERS; do
    echo "Benchmark n=$n workers=$p"
    # One warm-up run, excluded from CSV
    mpirun -np "$p" ./mpi_matmul "$n" "$SEED" >/dev/null

    for r in $(seq 1 "$REPS"); do
      line=$(mpirun -np "$p" ./mpi_matmul "$n" "$SEED")
      n_v=$(echo "$line" | sed -E 's/.*n=([0-9]+).*/\1/')
      p_v=$(echo "$line" | sed -E 's/.*workers=([0-9]+).*/\1/')
      setup=$(echo "$line" | sed -E 's/.*setup=([0-9.]+).*/\1/')
      bcast=$(echo "$line" | sed -E 's/.*bcast=([0-9.]+).*/\1/')
      scatter=$(echo "$line" | sed -E 's/.*scatter=([0-9.]+).*/\1/')
      compute=$(echo "$line" | sed -E 's/.*compute=([0-9.]+).*/\1/')
      gather=$(echo "$line" | sed -E 's/.*gather=([0-9.]+).*/\1/')
      comm=$(echo "$line" | sed -E 's/.*comm=([0-9.]+).*/\1/')
      total=$(echo "$line" | sed -E 's/.*total=([0-9.]+).*/\1/')
      checksum=$(echo "$line" | sed -E 's/.*checksum=([0-9]+).*/\1/')
      echo "$n_v,$p_v,$r,$setup,$bcast,$scatter,$compute,$gather,$comm,$total,$checksum" >> "$OUT"
    done
  done
done

echo "Saved $OUT"
