import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.optimize import curve_fit

# Suppress pandas warnings for cleaner console output
warnings.filterwarnings('ignore')

# Set your target file path here
file_path = '../cardia_run_log_2026-06-14_00-59-15.csv'

# ==========================================
# 1. ROBUST DATA EXTRACTION
# ==========================================
# Extract the instantaneous radial distribution from the VERY LAST row
with open(file_path, 'r') as f:
    lines = f.readlines()

last_line = lines[-1].strip()
radial_string = last_line.split(',')[-1]
# Get the radial bins for just the final tick to avoid temporal smearing
radial_probs = [float(val) for val in radial_string.split(';') if val]

# Load the main telemetry, ignoring the string column at the end
df = pd.read_csv(file_path, comment='#', usecols=range(34))

# Filter to active kinematic phase
# df = df[(df['Knot_Volume'] > 0) & (df['Sim_Tick'] >= 500)].copy()

df = df[(df['Sim_Tick'] >= 1350) & (df['Max_Tensor_Trace'] > 3.01)].copy()

ticks = df['Sim_Tick']

# ==========================================
# 2. PLOT 1: RADIAL PROBABILITY DENSITY
# ==========================================
r = np.linspace(0, 100, len(radial_probs))

# Safe normalization check
max_prob = np.max(radial_probs)
if max_prob > 1e-12:
    sim_normalized = radial_probs / max_prob
else:
    print("ℹ️ Shutter is still closed. Populating dummy step-function.")
    sim_normalized = np.zeros_like(r)
    sim_normalized[0:5] = 1.0

# Dynamically find the "Proton Surface" boundary
liftoff_threshold = 0.02
liftoff_indices = np.where(sim_normalized > liftoff_threshold)[0]
liftoff_idx = liftoff_indices[0] if len(liftoff_indices) > 0 else 0
r0 = r[liftoff_idx]

# Dynamically find the characteristic decay length
peak_index = np.argmax(sim_normalized)
r_peak = r[peak_index]
a0 = max(r_peak - r0, 1.0)

# Calculate Modified Finite-Nucleus Theory
theoretical = np.zeros_like(r)
for idx, radius in enumerate(r):
    if radius >= r0:
        r_eff = radius - r0
        theoretical[idx] = (r_eff**2) * np.exp(-2 * r_eff / a0)

max_theory = np.max(theoretical)
if max_theory > 1e-12:
    theoretical /= max_theory

# Calc Goodness-of-Fit Metrics
rmse = np.sqrt(np.mean((sim_normalized - theoretical) ** 2))
ss_res = np.sum((sim_normalized - theoretical) ** 2)
ss_tot = np.sum((sim_normalized - np.mean(sim_normalized)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

p = sim_normalized[liftoff_idx:] / np.sum(sim_normalized[liftoff_idx:])
q = theoretical[liftoff_idx:] / np.sum(theoretical[liftoff_idx:])
eps = 1e-12
p = np.clip(p, eps, None)
q = np.clip(q, eps, None)
kl_divergence = np.sum(p * np.log(p / q))

# Render Radial Plot
plt.figure(figsize=(10, 6))
plt.plot(r, sim_normalized, label=f'Simulated (Tick {df["Sim_Tick"].iloc[-1]})', color='#00ff00', linewidth=2.5)
plt.plot(r, theoretical, label=f'Extended Theory (r0={r0:.1f}, a0={a0:.1f})', linestyle='--', color='#ff00ff', linewidth=1.5)

metrics_box = (
    f"**Statistical Fit:**\n"
    f"  $R^2$ Score: {r_squared:.4f}\n"
    f"  RMSE: {rmse:.4f}\n"
    f"  KL Divergence: {kl_divergence:.4f}"
)
props = dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#e0e0e0', alpha=0.85)
plt.text(0.62, 0.15, metrics_box, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='bottom', bbox=props, fontfamily='monospace')

plt.title(f"Radial Probability Density Comparison at Tick {df['Sim_Tick'].iloc[-1]}")
plt.xlabel("Lattice Distance (voxels)")
plt.ylabel("Probability Density (Normalized)")
plt.grid(True, alpha=0.2)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# ==========================================
# 3. PLOT 2: ENGINE DIAGNOSTICS DASHBOARD
# ==========================================
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8
})

fig, axs = plt.subplots(2, 3, figsize=(16, 8), layout='constrained')
fig.suptitle("Cardia Engine: Renormalized Soliton Telemetry & Equilibrium", fontsize=14, fontweight='bold')

# Panel A
start_x, start_y, start_z = df['Center_X'].iloc[0], df['Center_Y'].iloc[0], df['Center_Z'].iloc[0]
axs[0, 0].plot(ticks, df['Center_X'] - start_x, label='X Drift', alpha=0.8)
axs[0, 0].plot(ticks, df['Center_Y'] - start_y, label='Y Drift', alpha=0.8)
axs[0, 0].plot(ticks, df['Center_Z'] - start_z, label='Z Drift', alpha=0.8)
axs[0, 0].set_title("A. Core Trajectory (Absolute Zitterbewegung)")
axs[0, 0].set_ylabel("Voxel Drift")
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend()

# Panel B
color1, color2 = 'purple', 'magenta'
axs[0, 1].set_title("B. Isolated Topological Spin & Circulation")
axs[0, 1].set_ylabel("Δ Helicity (LU)", color=color1)
line1 = axs[0, 1].plot(ticks, df['Total_Helicity'], label='Δ Helicity', color=color1, linewidth=2)
axs[0, 1].tick_params(axis='y', labelcolor=color1)

ax2 = axs[0, 1].twinx()
ax2.set_ylabel("Absolute Circulation", color=color2)
line2 = ax2.plot(ticks, df['Quantum_Circulation'], label='Circulation', color=color2, linestyle='--', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)

lines = line1 + line2
labels = [l.get_label() for l in lines]
axs[0, 1].legend(lines, labels, loc='lower right')
axs[0, 1].grid(True, alpha=0.3)

# Panel C
axs[0, 2].plot(ticks, df['Max_Tensor_Trace'], label='Δ Polymer Stretch', color='orange', linewidth=2)
axs[0, 2].axhline(y=4997, color='r', linestyle='--', label='Snap Limit (Δ L^2)')
axs[0, 2].set_title("C. FENE-P Tensor Stretch (Δ from Vacuum)")
axs[0, 2].set_ylabel("Δ Trace Value")
axs[0, 2].grid(True, alpha=0.3)
axs[0, 2].legend()

# Panel D
axs[1, 0].plot(ticks, df['Madelung_Pressure'], label='Δ Madelung (Quantum Potential)', color='blue')
axs[1, 0].plot(ticks, df['Elastic_Tension'], label='Δ Oldroyd-B Tension', color='orange')
axs[1, 0].plot(ticks, df['VC_Pressure'], label='Δ Vorticity Confinement', color='green')
axs[1, 0].set_title("D. Isolated Soliton Force Balance (Δ LU)")
axs[1, 0].set_xlabel("Simulation Tick")
axs[1, 0].set_ylabel("Force Magnitude (Δ LU)")
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].legend()

# Panel E
axs[1, 1].plot(ticks, df['Shape_Anisotropy'], label='Oblateness / Deformation', color='teal', linewidth=2)
axs[1, 1].set_title("E. Shape Anisotropy (Relativistic Deformation)")
axs[1, 1].set_xlabel("Simulation Tick")
axs[1, 1].set_ylabel("Anisotropy Ratio")
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].legend()

# Panel F
axs[1, 2].plot(ticks, df['Mach'], label='Peak Mach Number', color='red', linewidth=2)
axs[1, 2].axhline(y=0.4, color='k', linestyle='--', label='LBM Compressibility Limit')
axs[1, 2].set_title("F. Fluid Compressibility Check")
axs[1, 2].set_xlabel("Simulation Tick")
axs[1, 2].set_ylabel("Absolute Mach Number")
axs[1, 2].grid(True, alpha=0.3)
axs[1, 2].legend()

plt.show()

# ==========================================
# 4. PLOT 3: ANGULAR MOMENTUM EVOLUTION
# ==========================================
fig2, ax_ang = plt.subplots(figsize=(12, 5), layout='constrained')
fig2.suptitle("Quantum Spin State: Angular Momentum Transfer", fontsize=14, fontweight='bold')

ax_ang.plot(ticks, df['Ang_Mom_X'], label='$L_x$ (Off-axis Precession)', color='red', linewidth=2)
ax_ang.plot(ticks, df['Ang_Mom_Y'], label='$L_y$ (Off-axis Precession)', color='blue', linewidth=2)
ax_ang.set_ylabel("Off-Axis Momentum (LU)")
ax_ang.set_xlabel("Simulation Tick")
ax_ang.grid(True, alpha=0.3)

ax_ang_z = ax_ang.twinx()
ax_ang_z.plot(ticks, df['Ang_Mom_Z'], label='$L_z$ (Primary Spin)', color='green', linestyle='--', linewidth=2.5)
ax_ang_z.set_ylabel("Primary Z Momentum (LU)", color='green')

lines_ang = ax_ang.get_lines() + ax_ang_z.get_lines()
labels_ang = [l.get_label() for l in lines_ang]
ax_ang.legend(lines_ang, labels_ang, loc='center right')
plt.show()

# ==========================================
# 5. PLOT 4: VELOCITY VS. VACUUM DRAG
# ==========================================
# Safely Calculate Instantaneous Velocity (v_z) using shift differencing
df['v_z'] = (df['Center_Z'] - df['Center_Z'].shift(1)) / (df['Sim_Tick'] - df['Sim_Tick'].shift(1))
df['v_z_smooth'] = df['v_z'].rolling(window=20, center=True).mean()

# ✅ Isolate Drag Force directly from the Acoustic ZPE Radiation Pressure
df['F_drag_smooth'] = df['Casimir_Delta'].rolling(window=20, center=True).mean()

# Clean NaNs generated by shifts and rolls
df_clean = df.dropna(subset=['v_z_smooth', 'F_drag_smooth'])

v_abs = np.abs(df_clean['v_z_smooth'])
f_abs = np.abs(df_clean['F_drag_smooth'])

def drag_model(v, C, n):
    return C * (v ** n)

# Fit power-law model, looking for the n=2 ballistic dispersion
popt, _ = curve_fit(drag_model, v_abs, f_abs, bounds=(0, [np.inf, 3.0]))
C_fit, n_fit = popt

v_fit = np.linspace(v_abs.min(), v_abs.max(), 100)
f_fit = drag_model(v_fit, C_fit, n_fit)

plt.figure(figsize=(9, 6))
plt.scatter(v_abs, f_abs, color='purple', alpha=0.3, s=15, label='Ichor Acoustic Telemetry')
plt.plot(v_fit, f_fit, color='orange', linewidth=2.5,
         label=f'Acoustic Drag Fit\n$F = {C_fit:.2e} \\cdot v^{{{n_fit:.3f}}}$')

plt.title("Macroscopic Vacuum Drag: Acoustic Dispersion Analog", fontweight='bold')
plt.xlabel("Absolute Linear Velocity (Voxel / Tick)")
plt.ylabel("ZPE Radiation Pressure (Casimir Δ LU)")

plt.tight_layout()
plt.show()