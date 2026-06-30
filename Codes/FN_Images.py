import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# USER SETTINGS
# ==========================
results_dir = "/scratch/dnlrea/pedro.romano/results_S1_CNT75"
fields = [250, 500, 750, 1000]   # list of voltages / field labels used in filenames
output_dir = os.path.join(results_dir, "figures")
os.makedirs(output_dir, exist_ok=True)

AU_EFIELD_V_PER_M = 5.142206747e11  # atomic unit of E-field
BOHR_TO_ANG = 0.52917721092

# ==========================
# MATPLOTLIB ACS STYLE
# ==========================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.0,
    "lines.markersize": 7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "figure.dpi": 500
})

# ==========================
# LOAD SCALAR DATA
# ==========================
def load_scalar(fname):
    return float(np.loadtxt(fname))

E_applied = []
J_vals = []
sigma_vals = []
beta_vals = []

for f in fields:
    E_file = os.path.join(results_dir, f"beta_field_{f}.txt")
    J_file = os.path.join(results_dir, f"emission_current_{f}.txt")
    s_file = os.path.join(results_dir, f"conductivity_{f}.txt")

    beta = load_scalar(E_file)
    J = load_scalar(J_file)
    sigma = load_scalar(s_file)

    # Convert voltage label → field magnitude (if needed)
    # If you already know the field in V/m, replace this accordingly
    E_applied.append(f)  # keep as label or replace by actual field
    J_vals.append(J)
    sigma_vals.append(sigma)
    beta_vals.append(beta)

E_applied = np.array(E_applied, dtype=float)
J_vals = np.array(J_vals)
sigma_vals = np.array(sigma_vals)
beta_vals = np.array(beta_vals)

# ==========================
# 1️⃣ Emission current density plot
# ==========================
plt.figure(figsize=(4.5, 3.5))
plt.semilogy(E_applied, J_vals, "o-", label="CNT emission")

plt.xlabel("Applied field parameter")
plt.ylabel(r"Emission current density $J$ (A m$^{-2}$)")
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "J_vs_field.png"), dpi=600)
plt.close()

# ==========================
# 2️⃣ Effective conductivity plot
# ==========================
plt.figure(figsize=(4.5, 3.5))
plt.plot(E_applied, sigma_vals, "s-", color="red")

plt.xlabel("Applied field parameter")
plt.ylabel(r"Effective conductivity $\sigma$ (S m$^{-1}$)")
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "sigma_vs_field.png"), dpi=600)
plt.close()

# ==========================
# 3️⃣ Field enhancement factor plot
# ==========================
plt.figure(figsize=(4.5, 3.5))
plt.plot(E_applied, beta_vals, "d-", color="blue")

plt.xlabel("Applied field parameter")
plt.ylabel(r"Field enhancement factor $\beta$")
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "beta_vs_field.png"), dpi=600)
plt.close()

# ==========================
# 4️⃣ Fowler–Nordheim plot (VERY IMPORTANT)
# ==========================
# FN: ln(J / E^2) vs 1/E
# Replace E_applied with actual local field if available

E = E_applied.astype(float)
FN_x = 1.0 / E
FN_y = np.log(J_vals / E**2)

plt.figure(figsize=(4.5, 3.5))
plt.plot(FN_x, FN_y, "o-", color="green")

plt.xlabel(r"$1/E$")
plt.ylabel(r"$\ln(J/E^2)$")
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "FN_plot.png"), dpi=600)
plt.close()

# ==========================
# 5️⃣ Barrier profiles (one figure per field)
# ==========================
for f in fields:
    barrier_file = os.path.join(results_dir, f"barrier_field_{f}.txt")
    if not os.path.exists(barrier_file):
        continue

    data = np.loadtxt(barrier_file)
    x_bohr = data[:, 0]
    V = data[:, 1]

    x_ang = x_bohr * BOHR_TO_ANG

    plt.figure(figsize=(4.5, 3.5))
    plt.plot(x_ang, V, "-", color="black")

    plt.xlabel("Distance along emission direction (Å)")
    plt.ylabel("Electrostatic potential (Ha)")
    plt.title(f"Emission barrier (field = {f})", fontsize=11)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"barrier_field_{f}.png"),
        dpi=600
    )
    plt.close()

print("All ACS-quality figures generated successfully.")
