import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
EXECUTABLE = BUILD_DIR / "particle_filter_omp"

particle_counts = [1_000, 10_000, 100_000, 500_000]
thread_counts = [1, 2, 4, 8, 16]
repetitions = 5

results = []

for N in particle_counts:
    for p in thread_counts:
        runtimes = []

        print(f"\nN = {N:,}, threads = {p}")

        for run in range(1, repetitions + 1):
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(p)

            process = subprocess.run(
                [EXECUTABLE, str(N)],
                capture_output=True,
                text=True,
                env=env,
                check=True
            )

            match = re.search(r"Runtime:\s+([0-9.eE+-]+)", process.stdout)

            if not match:
                raise RuntimeError(
                    f"Could not find runtime in output:\n{process.stdout}"
                )

            runtime = float(match.group(1))
            runtimes.append(runtime)

            print(f"  run {run}: {runtime:.6f} s")

        median_runtime = np.median(runtimes)
        mean_runtime = np.mean(runtimes)
        std_runtime = np.std(runtimes, ddof=1)

        results.append({
            "N": N,
            "threads": p,
            "median_runtime": median_runtime,
            "mean_runtime": mean_runtime,
            "std_runtime": std_runtime
        })

        print(f"  median: {median_runtime:.6f} s")


df = pd.DataFrame(results)

# Calculate speedup and efficiency separately for each N
df["speedup"] = np.nan
df["efficiency"] = np.nan

for N in particle_counts:
    mask = df["N"] == N

    T1 = df.loc[
        mask & (df["threads"] == 1),
        "median_runtime"
    ].iloc[0]

    df.loc[mask, "speedup"] = (
        T1 / df.loc[mask, "median_runtime"]
    )

    df.loc[mask, "efficiency"] = (
        df.loc[mask, "speedup"]
        / df.loc[mask, "threads"]
    )

df.to_csv(f"{ROOT}/results/openmp_scaling.csv", index=False)

print("\nFinal results:")
print(df.to_string(index=False))
print("\nSaved: results/openmp_scaling.csv")