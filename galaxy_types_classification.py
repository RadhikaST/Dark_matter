import pandas as pd
import plotly.graph_objects as go
import glob
import os
from sklearn.linear_model import LinearRegression
import numpy as np

# Path to your folder with CSVs
folder_path = r"C:\Users\Radhika\OneDrive\Desktop\Dark matter\Dark_matter_data\Galaxy_csv_data"  # change to your folder path

# Load all CSV files
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

dataframes = []
for file in all_files:
    galaxy_name = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)

    # Make sure column names match your CSVs
    # If they are different, change 'radius_kpc' and 'velocity_kms' accordingly
    df = df.groupby('Rad', as_index=False)['Vobs'].mean()  # average duplicates
    df['galaxy'] = galaxy_name
    dataframes.append(df)

# Combine all data
data = pd.concat(dataframes, ignore_index=True)

# Train ML model on all galaxies
X = data[['Rad']]
y = data['Vobs']
model = LinearRegression()
model.fit(X, y)

# Predict and set threshold automatically
predicted = model.predict(X)
threshold = np.mean(predicted) + 2 * np.std(predicted)  # adjust factor for sensitivity

# Create Plotly figure
fig = go.Figure()

for galaxy in data['galaxy'].unique():
    galaxy_data = data[data['galaxy'] == galaxy]
    fig.add_trace(go.Scatter(
        x=galaxy_data['Rad'],
        y=galaxy_data['Vobs'],
        mode='lines+markers',
        name=galaxy,
        opacity=0.5,
        text=[f"Galaxy: {galaxy}<br>Radius: {r} kpc<br>Velocity: {v} km/s"
              for r, v in zip(galaxy_data['Rad'], galaxy_data['Vobs'])],
        hoverinfo='text'
    ))

# Add threshold line
fig.add_trace(go.Scatter(
    x=[data['Rad'].min(), data['Rad'].max()],
    y=[threshold, threshold],
    mode='lines',
    name='Threshold',
    opacity=1.0,
    line=dict(color='red', dash='dash')
))

# Layout settings
fig.update_layout(
    title="Galaxy Rotation Curves with outliers",
    xaxis_title="Radius (kpc)",
    yaxis_title="Velocity (km/s)",
    hovermode="closest",
    template="plotly_dark",
    showlegend=True  # can turn on if you want all 175 names
)

fig.show()
