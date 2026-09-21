import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("results/kalman_filter.csv")

# ============================================================
# some stats:
# ============================================================

rmse_obs = np.sqrt(np.mean((df["y_obs"] - df["x_true"])**2))
rmse_kf = np.sqrt(np.mean((df["m_post"] - df["x_true"])**2))
print(f"Observation RMSE: {rmse_obs:.6f}")
print(f"Kalman RMSE:      {rmse_kf:.6f}")

# ============================================================
# plot:
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(df["t"], df["x_true"], label="True state")
plt.plot(df["t"], df["y_obs"], label="Observation", alpha=0.4)
plt.plot(df["t"], df["m_post"], label="Kalman estimate")

plt.xlabel("Time")
plt.ylabel("State")
plt.title("Kalman Filter: True State vs Observation vs Estimate")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "kalman_filter.png",
    bbox_inches="tight",
    dpi=300
)

plt.savefig(
    FIGURES_DIR / "kalman_filter.pdf",
    bbox_inches="tight",
)

plt.show()
