from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_DIR / "openmp_runtime_breakdown.csv")

components = [
    "propagation_likelihood",
    "normalization",
    "posterior_mean",
    "resampling"
]

labels = [
    "Propagation + likelihood",
    "Normalization",
    "Posterior mean",
    "Resampling"
]


plt.figure(figsize=(8, 5.5))

bottom = pd.Series(0.0, index=df.index)

for component, label in zip(components, labels):
    plt.bar(
        df["threads"].astype(str),
        df[component],
        bottom=bottom,
        label=label
    )

    bottom += df[component]


plt.xlabel("Number of OpenMP threads")
plt.ylabel("Runtime (s)")
plt.title(r"Runtime decomposition for $N=500{,}000$")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "openmp_runtime_breakdown.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    FIGURES_DIR / "openmp_runtime_breakdown.pdf",
    bbox_inches="tight"
)

plt.show()