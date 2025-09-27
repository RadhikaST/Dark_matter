import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Configuration ---
input_dir = r"C:\Users\Radhika\OneDrive\Desktop\Dark matter\Dark_matter_data\Galaxy_csv_data"
output_file = r"C:\Users\Radhika\OneDrive\Desktop\Dark matter\Dark_matter_data\Outliers_sigma deviations\User input\SSummary_outliers.csv"
top_n = 10

# --- Ask user for sigma threshold ---
try:
    sigma_threshold = float(input("Enter sigma threshold (e.g., 2.5): "))
except:
    sigma_threshold = 2.5
    print("Invalid input. Using default 2.5σ.")

radius_grid = np.linspace(0.1, 20.0, 100)

# --- Load & Interpolate ---
files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
all_velocities = []
galaxy_names = []
interpolated_data = {}

for f in files:
    df = pd.read_csv(os.path.join(input_dir, f))
    name = f.replace(".csv", "")

    try:
        interp_func = interp1d(df["Rad"], df["Vobs"], kind='linear', bounds_error=False, fill_value='extrapolate')
        v_interp = interp_func(radius_grid)
        all_velocities.append(v_interp)
        galaxy_names.append(name)
        interpolated_data[name] = v_interp
    except:
        print(f"⚠️ Skipping {name} due to interpolation error.")

# --- Compute global statistics ---
vel_array = np.vstack(all_velocities)
mean_vel = np.mean(vel_array, axis=0)
std_vel = np.std(vel_array, axis=0)

# --- Evaluate & Summarize ---
summary_data = []

for i, name in enumerate(galaxy_names):
    v_obs = interpolated_data[name]
    sigma_deviation = np.abs(v_obs - mean_vel) / std_vel

    avg_obs_vel = np.mean(v_obs)
    avg_exp_vel = np.mean(mean_vel)
    avg_sigma = np.mean(sigma_deviation)
    avg_radius = np.mean(radius_grid)

    summary_data.append({
        "Galaxy Name": name,
        "Avg Radius (kpc)": round(avg_radius, 2),
        "Mean Obs Vel (km/s)": round(avg_obs_vel, 2),
        "Mean Exp Vel (km/s)": round(avg_exp_vel, 2),
        "Mean Sigma Dev": round(avg_sigma, 3),
        "Outlier?": "Yes" if avg_sigma > sigma_threshold else "No"
    })

# --- Rank & Save ---
df_summary = pd.DataFrame(summary_data)
df_summary.sort_values("Mean Sigma Dev", ascending=False, inplace=True)
df_summary.reset_index(drop=True, inplace=True)
df_summary.to_csv(output_file, index=False)

print(f"\n Summary saved to: {output_file}")
print(f" Top {top_n} Anomalous Galaxies:\n")
print(df_summary.head(top_n)[["Galaxy Name", "Mean Sigma Dev"]])

# --- Plots ---
plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(df_summary["Mean Sigma Dev"], bins=25, color='skyblue', edgecolor='black')
plt.axvline(sigma_threshold, color='red', linestyle='--', label=f'{sigma_threshold}σ Threshold')
plt.title("Histogram of Mean Sigma Deviations")
plt.xlabel("Mean Sigma Deviation")
plt.ylabel("Galaxy Count")
plt.legend()

# Boxplot
plt.subplot(1, 2, 2)
plt.boxplot(df_summary["Mean Sigma Dev"], vert=False)
plt.title("Boxplot of Mean Sigma Deviations")
plt.xlabel("Mean Sigma Deviation")

plt.tight_layout()
plt.show()
