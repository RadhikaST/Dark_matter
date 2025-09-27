import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

input_dir = r"C:\Users\Radhika\OneDrive\Desktop\Dark matter\Dark_matter_data\Galaxy_csv_data"
output_file = r"C:\Users\Radhika\OneDrive\Desktop\Dark matter\Dark_matter_data\Outliers_sigma deviations\User input\Outliers.csv"  # change path if needed

files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

radius_grid = np.linspace(0.1, 20.0, 100)  # Standard radius grid (in kpc)

all_velocities = []
galaxy_names = []
interpolated_data = {}

# 1. Interpolate each galaxy to standard radius grid
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
        print(f"Skipping {name} due to interpolation error")

# 2. Compute mean and std deviation of velocity at each radius
vel_array = np.vstack(all_velocities)
mean_vel = np.mean(vel_array, axis=0)
std_vel = np.std(vel_array, axis=0)

# 3. Identify outliers and collect detailed data
outliers = []
outlier_rows = []

for i, name in enumerate(galaxy_names):
    v_obs = interpolated_data[name]
    deviations = np.abs(v_obs - mean_vel)
    z_scores = deviations / std_vel

    if np.any(z_scores > 3.0):
        outliers.append(name)

        # Save only the points where galaxy is an outlier
        for j, z in enumerate(z_scores):
            if z > 2.0:
                outlier_rows.append({
                    "Galaxy Name": name,
                    "Radius (kpc)": radius_grid[j],
                    "Observed Velocity (km/s)": v_obs[j],
                    "Mean Velocity (km/s)": mean_vel[j],
                    "Sigma Deviation": z
                })

# 4. Output summary to console
print("🔴 Outlier Galaxies Detected:")
for g in outliers:
    print("-", g)

# 5. Save detailed outlier data to CSV
df_outliers = pd.DataFrame(outlier_rows)
df_outliers.to_csv(output_file, index=False)
print(f"\n✅ Outlier details saved to: {output_file}")

