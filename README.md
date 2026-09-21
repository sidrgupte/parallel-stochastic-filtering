# Parallel Stochastic Filtering

A progressive study of stochastic filtering through the lens of parallel computing.

The project starts from a setting where the filtering problem is exactly solvable, builds a particle-filtering implementation from that baseline, and then studies where parallelism helps, and where it does not.

The codebase is written primarily in **C++23**, with **OpenMP** for shared-memory parallelism and Python for analysis and visualization.

---

## Stage I - Linear-Gaussian Filtering with OpenMP

The first stage uses a scalar linear-Gaussian state-space model,

**Xₜ = 0.9 Xₜ₋₁ + 0.5 εₜ, Yₜ = Xₜ + ηₜ**

with an exact Kalman filter as the numerical benchmark.

A bootstrap particle filter is then implemented and parallelized with OpenMP.

### Main results

- The particle filter converges toward the Kalman solution at approximately the expected Monte Carlo rate,

  **error ∝ N⁻¹ᐟ²**

  with an empirical log-log slope of approximately **-0.497**.

- Particle propagation and likelihood evaluation scale well with OpenMP. At **N = 500,000**, the kernel achieved about **6.8× speedup on 16 threads**.

- End-to-end scaling is much weaker, reaching only about **1.15×**, because multinomial resampling remains serial and eventually dominates the runtime.

- A synchronization experiment compares an unsafe shared accumulation, an OpenMP `critical` section, and an OpenMP `reduction`. The reduction preserves correctness without serializing every update.

---

## Repository

```text
.
├── src/        C++ filtering implementations
├── scripts/    experiments, benchmarks, and plotting
├── results/    compact numerical benchmark data
└── report/     LaTeX report and final figures
```

The project is being developed in stages. Stage I is complete; later stages will extend both the filtering model and the parallel implementation.

---

## Build

```bash
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

The main Stage-I executables are:

```text
stochastic_filter
kalman_filter
particle_filter_serial
particle_filter_omp
```

For example,

```bash
OMP_NUM_THREADS=8 ./particle_filter_omp 100000
```

runs the OpenMP particle filter with 8 threads and 100,000 particles.

---

## Report

The accompanying report contains the numerical results, OpenMP experiments, and performance analysis:

**[Parallel Stochastic Filtering - Current Report](report/main.pdf)**

The report is updated as each stage is completed.

---

## Status

| Stage | Status |
|---|---|
| I - SLinear-Gaussian benchmark and OpenMP particle filter | **Complete** |
| II | In development |
| III | Planned |
| IV | Planned |
| V | Planned |