import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import plotly.io as pio
import webbrowser

# CONFIG
DATA_DIR = r"C:\Users\Radhika\OneDrive\Desktop\Dark matter\Dark_matter_data\Galaxy_csv_data"
PLOTLY_TEMPLATE = "plotly_white"
pio.renderers.default = "browser"

G = 6.67430e-11
KPC_TO_M = 3.085677581e19
KM_S_TO_M_S = 1e3
MSUN = 1.98847e30

# HELPERS
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
    except:
        rho0_fit, rs_fit = np.nan, np.nan

    v_dm_fit = nfw_vcirc(r_kpc, rho0_fit, rs_fit) if np.isfinite(rho0_fit) else np.zeros_like(r_kpc)
    v_total_fit = np.sqrt(np.maximum(v_bary_kms**2 + v_dm_fit**2, 0.0))

    r_out = float(r_kpc[-1])
    vobs_out = float(vobs_kms[-1])
    vbary_out = float(v_bary_kms[-1])
    vdm_out = float(v_dm_fit[-1]) if np.isfinite(rho0_fit) else 0.0
    vmodel_out = float(v_total_fit[-1])

    M_expected = enclosed_mass_from_vr(vobs_out, r_out)
    M_visible = enclosed_mass_from_vr(vbary_out, r_out)
    M_dark = max(M_expected - M_visible, 0.0)
    dm_fraction = M_dark / M_expected if M_expected > 0 else np.nan

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

# PROCESS GALAXIES
csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
results = {os.path.splitext(os.path.basename(f))[0]: process_galaxy(f) for f in csv_files}
first_galaxy = list(results.keys())[0]
html_file = "interactive_galaxy_mobile_fixed.html"

# PREPARE JS DATA
galaxy_data_js = {}
for gname, (gn, r, vobs, vbary, vdm, vtot, summary) in results.items():
    galaxy_data_js[gname] = {
        "r": r.tolist(),
        "vobs": vobs.tolist(),
        "vbary": vbary.tolist(),
        "vdm": vdm.tolist(),
        "vtot": vtot.tolist(),
        "summary": summary
    }

html_content = f"""
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
<h3>Select Galaxy:</h3>
<select id="galaxySelect">
{"".join([f'<option value="{g}">{g}</option>' for g in results.keys()])}
</select>
<div id="plotDiv" style="width:100%;height:80vh;"></div>

<script>
const galaxyData = {galaxy_data_js};

function makeTraces(gname) {{
    const data = galaxyData[gname];
    return [
        {{x:data.r, y:data.vobs, mode:"markers", name:"Observed Vobs"}},
        {{x:data.r, y:data.vbary, mode:"lines", name:"Baryons only"}},
        {{x:data.r, y:data.vdm, mode:"lines", name:"NFW DM only"}},
        {{x:data.r, y:data.vtot, mode:"lines", name:"Total model"}},
        {{x:[0], y:[0], text:[data.summary], mode:"text", showlegend:false, xaxis:'x2', yaxis:'y2'}}
    ];
}}

function makeLayout(gname) {{
    return {{
        title: gname + " — Rotation Curve & NFW Fit",
        template: "{PLOTLY_TEMPLATE}",
        grid: {{rows:1, columns:2, pattern:'independent'}},
        xaxis: {{title:"Radius (kpc)"}},
        yaxis: {{title:"Velocity (km/s)"}},
        xaxis2: {{showticklabels:false, domain:[1.5,0.25]}},
        yaxis2: {{showticklabels:false, domain:[0,1]}},
        height:600,
        responsive:true
    }};
}}

Plotly.newPlot('plotDiv', makeTraces("{first_galaxy}"), makeLayout("{first_galaxy}"), {{responsive:true}});

document.getElementById("galaxySelect").addEventListener("change", function(){{
    const gname = this.value;
    Plotly.react('plotDiv', makeTraces(gname), makeLayout(gname));
}});
</script>
</body>
</html>
"""

with open(html_file, "w", encoding="utf-8") as f:
    f.write(html_content)

webbrowser.open(f"file://{os.path.abspath(html_file)}")
