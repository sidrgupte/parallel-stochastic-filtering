from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INPUT_FILE = RESULTS_DIR / "openmp_scaling.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
df = df.sort_values(["N", "threads"]).reset_index(drop=True)

# ------------------------------------------------------------
# Recompute performance metrics from the measured median runtime
# ------------------------------------------------------------

df["speedup"] = 0.0
df["efficiency"] = 0.0

for N in sorted(df["N"].unique()):
    mask = df["N"] == N

    T1 = df.loc[
        mask & (df["threads"] == 1),
        "median_runtime"
    ].iloc[0]

    df.loc[mask, "speedup"] = T1 / df.loc[mask, "median_runtime"]
    df.loc[mask, "efficiency"] = df.loc[mask, "speedup"] / df.loc[mask, "threads"]


particle_counts = sorted(df["N"].unique())
thread_counts = sorted(df["threads"].unique())


# ============================================================
# Figure 1: Strong-scaling speedup
# ============================================================

plt.figure(figsize=(8, 5.5))

for N in particle_counts:
    d = df[df["N"] == N]

    plt.plot(
        d["threads"],
        d["speedup"],
        marker="o",
        linewidth=1.8,
        markersize=6,
        label=f"N = {N:,}"
    )

# Speedup = 1 means parallel execution is no faster than serial.
plt.axhline(1.0, linestyle="--", linewidth=1.2, label="No speedup")

plt.xscale("log", base=2)
plt.xticks(thread_counts, labels=thread_counts)

plt.xlabel("Number of OpenMP threads")
plt.ylabel(r"Speedup $S_p = T_1/T_p$")
plt.title("Strong-scaling speedup of the particle filter")

plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "openmp_speedup.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    FIGURES_DIR / "openmp_speedup.pdf",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Figure 2: Parallel efficiency
# ============================================================

plt.figure(figsize=(8, 5.5))

for N in particle_counts:
    d = df[df["N"] == N]

    plt.plot(
        d["threads"],
        d["efficiency"],
        marker="o",
        linewidth=1.8,
        markersize=6,
        label=f"N = {N:,}"
    )

plt.xscale("log", base=2)
plt.xticks(thread_counts, labels=thread_counts)

plt.ylim(0.0, 1.05)

plt.xlabel("Number of OpenMP threads")
plt.ylabel(r"Parallel efficiency $E_p = S_p/p$")
plt.title("Parallel efficiency of the particle filter")

plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "openmp_efficiency.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    FIGURES_DIR / "openmp_efficiency.pdf",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Print final scaling table
# ============================================================

summary = df[
    [
        "N",
        "threads",
        "median_runtime",
        "speedup",
        "efficiency"
    ]
].copy()

print("\nOpenMP scaling summary\n")
print(
    summary.to_string(
        index=False,
        formatters={
            "median_runtime": lambda x: f"{x:.6f}",
            "speedup": lambda x: f"{x:.3f}",
            "efficiency": lambda x: f"{x:.3f}"
        }
    )
)

print(f"\nRead:   {INPUT_FILE}")
print(f"Saved:  {FIGURES_DIR / 'openmp_speedup.png'}")
print(f"Saved:  {FIGURES_DIR / 'openmp_speedup.pdf'}")
print(f"Saved:  {FIGURES_DIR / 'openmp_efficiency.png'}")
print(f"Saved:  {FIGURES_DIR / 'openmp_efficiency.pdf'}")