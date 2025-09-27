import os
import pandas as pd
import plotly.graph_objects as go

# ===> CHANGE THIS to the path where your cleaned CSV files are stored
data_folder = r"C:\Users\Radhika\OneDrive\Desktop\Dark matter\Dark_matter_data\Galaxy_csv_data"

fig = go.Figure()

for file in os.listdir(data_folder):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_folder, file))
        galaxy_name = file.replace('.csv', '')

        fig.add_trace(go.Scatter(
            x=df['Rad'],
            y=df['Vobs'],
            mode='lines+markers',
            name=galaxy_name,
            hovertemplate=(
                f"<b>{galaxy_name}</b><br>" +
                "Radius: %{x}<br>Velocity: %{y}<extra></extra>"
            )
        ))

fig.update_layout(
    title="Observed Rotation Curves for 175 Galaxies",
    xaxis_title="Radius (kpc)",
    yaxis_title="Observed Velocity (km/s)",
    hovermode='closest',
    width=1200,
    height=800
)

fig.show()
