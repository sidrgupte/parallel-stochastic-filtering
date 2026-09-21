import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Read exact Kalman benchmark
kf = pd.read_csv("results/kalman_filter.csv")

results = []

# Read all particle-filter runs
for filename in glob.glob("results/particle_filter_serial_N*.csv"):
    match = re.search(r"N(\d+)\.csv$", filename)

    if not match:
        continue

    N = int(match.group(1))
    pf = pd.read_csv(filename)

    rmse_pf_kf = np.sqrt(np.mean((pf["pf_mean"] - kf["m_post"])**2))
    rmse_pf_true = np.sqrt(np.mean((pf["pf_mean"] - pf["x_true"])**2))

    results.append((N, rmse_pf_kf, rmse_pf_true))

results.sort()

# Store results
df = pd.DataFrame(results, columns=["N", "pf_kf_rmse", "pf_true_rmse"])
df.to_csv("results/pf_convergence.csv", index=False)

# Print results
for N, rmse_pf_kf, rmse_pf_true in results:
    print(
        f"N = {N:>7}, "
        f"PF-KF RMSE = {rmse_pf_kf:.6f}, "
        f"PF-Truth RMSE = {rmse_pf_true:.6f}"
    )

# Log-log convergence slope
coeff = np.polyfit(np.log(df["N"]), np.log(df["pf_kf_rmse"]), 1)
slope = coeff[0]

print(f"Log-log slope: {slope:.3f}")

# N^{-1/2} reference curve, anchored at the first observation
reference = df["pf_kf_rmse"].iloc[0] * np.sqrt(df["N"].iloc[0] / df["N"])

# Plot
plt.figure(figsize=(10, 6))

plt.plot(
    df["N"],
    df["pf_kf_rmse"],
    marker="o",
    label="PF-KF RMSE"
)

plt.plot(
    df["N"],
    reference,
    linestyle="--",
    label=r"$N^{-1/2}$ reference"
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Number of particles $N$")
plt.ylabel("PF-KF RMSE")
plt.title("Bootstrap Particle Filter Convergence to Kalman Filter")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "serial_pf_convergence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    FIGURES_DIR / "serial_pf_convergence.pdf",
    bbox_inches="tight"
)

plt.show()