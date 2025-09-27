import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# =========================
# CONFIG
# =========================
DATA_DIR = r"C:\Users\Radhika\OneDrive\Desktop\Dark matter\Dark_matter_data\Galaxy_csv_data"
PLOTLY_TEMPLATE = "plotly_white"  # or "plotly_dark"
pio.renderers.default = "browser"  # <--- FIX: Always open in browser

# Physical constants & unit conversions
G = 6.67430e-11                 # m^3 kg^-1 s^-2
KPC_TO_M = 3.085677581e19       # meters
KM_S_TO_M_S = 1e3               # km/s -> m/s
MSUN = 1.98847e30               # kg

# =========================
# HELPERS
# =========================
def enclosed_mass_from_vr(v_kms, r_kpc):
    v = np.asarray(v_kms) * KM_S_TO_M_S
    r = np.asarray(r_kpc) * KPC_TO_M
    M = (v**2 * r) / G
    return M / MSUN

def nfw_vcirc(r_kpc, rho0, rs_kpc):
    r = np.asarray(r_kpc) * KPC_TO_M
    rs = rs_kpc * KPC_TO_M
    x = np.clip(r / rs, 1e-6, None)
    M_dm = 4.0 * np.pi * rho0 * (rs**3) * (np.log(1.0 + x) - (x / (1.0 + x)))
    v2 = G * M_dm / r
    return np.sqrt(v2) / KM_S_TO_M_S

def baryonic_velocity(df):
    terms = []
    for col in ["Vgas", "Vdisk", "Vbul"]:
        if col in df.columns:
            terms.append(np.square(df[col].to_numpy(dtype=float)))
    if not terms:
        return np.zeros(len(df), dtype=float)
    return np.sqrt(np.sum(terms, axis=0))

def process_galaxy(csv_path):
    galaxy_name = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path)

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Rad", "Vobs"])
    df = df.sort_values("Rad")
    r_kpc = df["Rad"].to_numpy(dtype=float)
    vobs_kms = df["Vobs"].to_numpy(dtype=float)

    if "errV" in df.columns:
        verr_kms = np.clip(df["errV"].to_numpy(dtype=float), 5.0, None)
    else:
        verr_kms = np.full_like(vobs_kms, 10.0)

    v_bary_kms = baryonic_velocity(df)

    def v_model_kms(r_kpc, rho0, rs_kpc):
        v_dm = nfw_vcirc(r_kpc, rho0, rs_kpc)
        return np.sqrt(np.maximum(v_bary_kms**2 + v_dm**2, 0.0))

    p0 = [1e-23, 10.0]
    bounds = ([1e-26, 0.5], [1e-20, 100.0])

    try:
        popt, _ = curve_fit(
            lambda r, rho0, rs: v_model_kms(r, rho0, rs),
            r_kpc, vobs_kms, sigma=verr_kms, absolute_sigma=True,
            p0=p0, bounds=bounds, maxfev=20000
        )
        rho0_fit, rs_fit = popt
    except Exception as e:
        rho0_fit, rs_fit = np.nan, np.nan
        print(f"⚠️ Fit failed for {galaxy_name}: {e}")

    v_dm_fit = nfw_vcirc(r_kpc, rho0_fit, rs_fit) if np.isfinite(rho0_fit) else np.zeros_like(r_kpc)
    v_total_fit = np.sqrt(np.maximum(v_bary_kms**2 + v_dm_fit**2, 0.0))

    # Calculations
    r_out = float(r_kpc[-1])
    vobs_out = float(vobs_kms[-1])
    vbary_out = float(v_bary_kms[-1])
    vdm_out = float(v_dm_fit[-1]) if np.isfinite(rho0_fit) else 0.0
    vmodel_out = float(v_total_fit[-1])

    M_expected = enclosed_mass_from_vr(vobs_out, r_out)
    M_visible = enclosed_mass_from_vr(vbary_out, r_out)
    M_dark = max(M_expected - M_visible, 0.0)
    dm_fraction = M_dark / M_expected if M_expected > 0 else np.nan

    # Text summary
    summary_text = (
        f"<b>Galaxy:</b> {galaxy_name}<br>"
        f"Data points: {len(df)}<br>"
        f"Outer radius: {r_out:.2f} kpc<br>"
        f"Observed V: {vobs_out:.1f} km/s<br>"
        f"Baryonic V: {vbary_out:.1f} km/s<br>"
        f"NFW V: {vdm_out:.1f} km/s<br>"
        f"Model V: {vmodel_out:.1f} km/s<br>"
        f"M_expected: {M_expected:.3e} Msun<br>"
        f"M_visible: {M_visible:.3e} Msun<br>"
        f"M_dark: {M_dark:.3e} Msun<br>"
        f"Dark matter %: {dm_fraction*100:.2f}%<br>"
        f"rho0: {rho0_fit:.3e} kg/m³<br>"
        f"rs: {rs_fit:.2f} kpc"
    )

    return galaxy_name, r_kpc, vobs_kms, v_bary_kms, v_dm_fit, v_total_fit, summary_text

# =========================
# MAIN INTERACTIVE PLOT
# =========================
csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
results = {os.path.splitext(os.path.basename(f))[0]: process_galaxy(f) for f in csv_files}

first_galaxy = list(results.keys())[0]
_, r0, vobs0, vbary0, vdm0, vtot0, summary0 = results[first_galaxy]

fig = make_subplots(
    rows=1, cols=2, column_widths=[0.7, 0.3],
    subplot_titles=("Rotation Curve", "Galaxy Data")
)

# Left: rotation curves
fig.add_trace(go.Scatter(x=r0, y=vobs0, mode="markers", name="Observed Vobs"), row=1, col=1)
fig.add_trace(go.Scatter(x=r0, y=vbary0, mode="lines", name="Baryons only"), row=1, col=1)
fig.add_trace(go.Scatter(x=r0, y=vdm0, mode="lines", name="NFW DM only"), row=1, col=1)
fig.add_trace(go.Scatter(x=r0, y=vtot0, mode="lines", name="Total model"), row=1, col=1)

# Right: summary text
fig.add_trace(go.Scatter(
    x=[0], y=[0], text=[summary0],
    mode="text", showlegend=False
), row=1, col=2)

# Dropdown menu
dropdown_buttons = []
for gname, (gn, r, vobs, vbary, vdm, vtot, summary) in results.items():
    dropdown_buttons.append(dict(
        label=gn,
        method="update",
        args=[
            {
                "x": [r, r, r, r, [0]],
                "y": [vobs, vbary, vdm, vtot, [0]],
                "text": [None, None, None, None, [summary]],
            },
            {"title": f"{gn} — Rotation Curve & NFW Fit"}
        ]
    ))

fig.update_layout(
    template=PLOTLY_TEMPLATE,
    height=600, width=1000,
    title=f"{first_galaxy} — Rotation Curve & NFW Fit",
    updatemenus=[dict(
        active=0, buttons=dropdown_buttons,
        x=0.45, y=1.15, xanchor="center", yanchor="top"
    )]
)

fig.update_xaxes(title_text="Radius (kpc)", row=1, col=1)
fig.update_yaxes(title_text="Velocity (km/s)", row=1, col=1)
fig.write_html(
    "interactive_galaxy_pc.html",
    full_html=True,
    include_plotlyjs='cdn',
    config={
        'responsive': True,   # responsive layout
        'scrollZoom': True,   # pinch-to-zoom on mobile
        'displayModeBar': True
    }
)


fig.show()

