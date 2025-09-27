import os
import pandas as pd

# ===> CHANGE THIS to the folder where your .dat files are stored
input_folder = r"C:\Users\Radhika\OneDrive\Desktop\Dark matter\Dark_matter_data\Galaxy_rotmod_files"

# ===> CHANGE THIS to where you want the cleaned CSV files to go
output_folder = r"C:\Users\Radhika\OneDrive\Desktop\Dark matter\Dark_matter_data\Galaxy_csv_data"
os.makedirs(output_folder, exist_ok=True)

columns = ['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']

for filename in os.listdir(input_folder):
    if filename.endswith(".dat"):
        with open(os.path.join(input_folder, filename), 'r') as f:
            lines = f.readlines()

        clean_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        data = [line.split() for line in clean_lines if len(line.split()) == len(columns)]

        if data:
            df = pd.DataFrame(data, columns=columns).astype(float)
            output_file = os.path.join(output_folder, filename.replace(".dat", ".csv"))
            df.to_csv(output_file, index=False)

print("All files converted successfully.")
