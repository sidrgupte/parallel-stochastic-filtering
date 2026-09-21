import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

kf = pd.read_csv("results/kalman_filter.csv")
pf = pd.read_csv("results/particle_filter_serial.csv")

rmse_kf_true = np.sqrt(np.mean((kf["m_post"] - kf["x_true"])**2))
rmse_pf_true = np.sqrt(np.mean((pf["pf_mean"] - pf["x_true"])**2))
rmse_pf_kf = np.sqrt(np.mean((pf["pf_mean"] - kf["m_post"])**2))

print(f"Kalman vs truth RMSE: {rmse_kf_true:.6f}")
print(f"PF vs truth RMSE:     {rmse_pf_true:.6f}")
print(f"PF vs Kalman RMSE:    {rmse_pf_kf:.6f}")

plt.figure(figsize=(12, 6))

plt.plot(kf["t"], kf["x_true"], label="True state")
plt.plot(kf["t"], kf["m_post"], label="Kalman filter")
plt.plot(pf["t"], pf["pf_mean"], label="Particle filter", linestyle="--")

plt.xlabel("Time")
plt.ylabel("State")
plt.title("Kalman Filter vs Bootstrap Particle Filter")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()