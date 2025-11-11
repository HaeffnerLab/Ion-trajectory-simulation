import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 16-term polynomial V(x,y,z) from the expansion in the header comments
# Note: works with NumPy arrays for y

def V_poly(x, y, z):
    return (
        2.3777691012078492e-17
        + 1.9165482389643225e-20 * z
        + 3.8917048378339124e-17 * y
        - 1.2946583820318695e-20 * x
        - 1.1431455592215858e-15 * z**2
        - 1.9670963026343986e-20 * y * z
        + 2.2832634893514657e-15 * y**2
        - 5.789551794809683e-25 * x * z
        + 4.1678133287307315e-21 * x * y
        - 1.1401179301298797e-15 * x**2
        - 1.8936614072723407e-13 * y * z**2
        + 1.2601027689595814e-13 * y**3
        - 1.7777849403692365e-19 * x * z**2
        + 5.97764571403142e-25 * x * y * z
        + 7.107425557844035e-19 * x * y**2
        - 1.8866468996064039e-13 * x**2 * y
        - 1.7765468724915995e-19 * x**3
        + 4.72249446479168e-12 * z**4
        + 6.597307394847976e-32 * y * z**3
        - 3.770994386852272e-11 * y**2 * z**2
        - 3.1130232580258706e-31 * y**3 * z
        + 1.2500016623929169e-11 * y**4
        - 1.1217044340063144e-37 * x * z**3
        - 2.4896484718686223e-17 * x * y * z**2
        + 7.510570436377456e-38 * x * y**2 * z
        + 3.3212324470581736e-17 * x * y**3
        + 9.374977079772627e-12 * x**2 * z**2
        + 7.3598775556232195e-31 * x**2 * y * z
        - 3.7290155875052295e-11 * x**2 * y**2
        + 8.713520861270657e-38 * x**3 * z
        - 2.491349623101966e-17 * x**3 * y
        + 4.652529799213278e-12 * x**4
        + 1.9113196384177776e-10 * y * z**4
        - 5.013631479153249e-10 * y**3 * z**2
        + 9.528261657225398e-11 * y**5
        + 1.4789653914147206e-15 * x * z**4
        - 1.7741494003809537e-14 * x * y**2 * z**2
        + 1.1819545079871444e-14 * x * y**4
        + 3.5729766069530865e-10 * x**2 * y * z**2
        - 4.51463017807215e-10 * x**2 * y**3
        + 2.955900551773739e-15 * x**3 * z**2
        - 1.772525882513971e-14 * x**3 * y**2
        + 1.6618189878772271e-10 * x**4 * y
        + 1.476935827336597e-15 * x**5
        - 3.217225121477833e-10 * z**6
        + 5.145669821674156e-9 * y**2 * z**4
        - 5.141525659544197e-9 * y**4 * z**2
        - 1.193336166486265e-12 * y**6
        - 1.0079717404981153e-15 * x * y * z**4
        + 2.8422455870555285e-15 * x * y**3 * z**2
        - 1.8538798290880499e-16 * x * y**5
        - 3.1983213945739613e-10 * x**2 * z**4
        - 2.4864972779747342e-11 * x**2 * y**2 * z**2
        + 5.159425702041491e-9 * x**2 * y**4
        - 8.263021060592981e-16 * x**3 * y * z**2
        - 3.2945525265582636e-16 * x**3 * y**3
        + 3.2397630158735423e-10 * x**4 * z**2
        - 5.155281539911533e-9 * x**4 * y**2
        + 1.8146678640267772e-16 * x**5 * y
        + 3.220870158882786e-10 * x**6
        - 3.88472104575678e-11 * y * z**6
        + 9.104572254373486e-11 * y**3 * z**4
        + 6.145265651846e-11 * y**5 * z**2
        + 5.004982104870894e-15 * y**7
        + 9.498433576059158e-17 * x * z**6
        - 2.051493901102545e-15 * x * y**2 * z**4
        + 3.0384132379248263e-15 * x * y**4 * z**2
        + 1.0220695094267911e-18 * x * y**6
        + 3.095709892323125e-10 * x**2 * y * z**4
        - 1.1608009004470089e-9 * x**2 * y**3 * z**2
        - 6.155776114266231e-11 * x**2 * y**5
        + 2.089096215645571e-16 * x**3 * z**4
        - 1.9738386736445622e-15 * x**3 * y**2 * z**2
        - 1.0179147601887426e-15 * x**3 * y**4
        + 2.70829460991192e-10 * x**4 * y * z**2
        + 2.96063085312272e-10 * x**4 * y**3
        + 7.203809442572191e-17 * x**5 * z**2
        + 8.081327234777017e-16 * x**5 * y**2
        - 7.726791446186718e-11 * x**6 * y
        - 4.1912896090639234e-17 * x**7
        + 8.2242222510736e-13 * z**8
        - 1.0677769902825629e-11 * y**2 * z**6
        - 3.2500161850676465e-12 * y**4 * z**4
        - 1.8114891918553227e-13 * y**6 * z**2
        - 7.16700970029145e-18 * y**8
        - 1.3251901862094667e-14 * x * y * z**6
        + 5.2980879056353287e-14 * x * y**3 * z**4
        - 1.2275328881558064e-17 * x * y**5 * z**2
        - 1.7757126344847943e-21 * x * y**7
        - 1.235005240018021e-11 * x**2 * z**6
        + 1.796666456527904e-10 * x**2 * y**2 * z**4
        + 2.2217330898188936e-11 * x**2 * y**4 * z**2
        + 1.8134959545714427e-13 * x**2 * y**6
        + 1.3278630254120066e-14 * x**3 * y * z**4
        - 1.0592084034976807e-13 * x**3 * y**3 * z**2
        + 4.104206282296202e-18 * x**3 * y**5
        + 9.306900583188096e-13 * x**4 * z**4
        - 2.0188397655097932e-10 * x**4 * y**2 * z**2
        - 4.156262471674348e-12 * x**4 * y**4
        + 2.3809073952458383e-14 * x**5 * y * z**2
        + 1.0587979828694512e-14 * x**5 * y**3
        + 1.308665574673777e-11 * x**6 * z**2
        + 1.5121436758735026e-11 * x**6 * y**2
        - 2.6463339732639004e-15 * x**7 * y
        - 1.0074318751954571e-12 * x**8
    )

# Evaluation parameters
x0 = 0.0
z0 = 0.0
# y-range in micrometers
y_um = np.linspace(-18.0, 100.0, 1000)

# Evaluate the full 16-term polynomial at x=0, z=0
V = V_poly(x0, y_um, z0)

# Plot
plt.figure(figsize=(7, 4.5))
plt.title('Fitted Potential')
plt.plot(y_um, V, linewidth=2, label="polynomial")
plt.legend()
# === Overlay raw data from CSV at x≈0, z≈0 ===
raw_csv = "b=2.52a y(-68,200).csv"

# Try reading with potential COMSOL header rows; fallback without skiprows
try:
    df = pd.read_csv(raw_csv, skiprows=8)
except Exception:
    df = pd.read_csv(raw_csv)

# Helper to pick a column by keywords
def _pick_col(_df, keywords):
    for c in _df.columns:
        cl = c.strip().lower()
        if any(k in cl for k in keywords):
            return c
    return None

# Detect columns (handles names like "% x", "y", "z", and "...V (V)")
xcol = _pick_col(df, ["% x", " x", "x"])  # try to avoid matching inside longer names first
ycol = _pick_col(df, [" y", "y"])  # prefer exact/space-prefixed
zcol = _pick_col(df, [" z", "z"])  # may be None for 2D data
Vcol = _pick_col(df, [" v (v", "(v", "potential", ".v (v", " v)"])
if Vcol is None:
    Vcol = df.columns[-1]  # fallback to last column

# Coerce numeric and drop NaNs
for c in [xcol, ycol] + ([zcol] if zcol else []):
    df[c] = pd.to_numeric(df[c], errors='coerce')
df[Vcol] = pd.to_numeric(df[Vcol], errors='coerce')
df = df.dropna(subset=[xcol, ycol, Vcol] + ([zcol] if zcol else []))

# Select the slice at or nearest to x=0, then z=0
x_target = 0.0
x_sel = df.loc[(df[xcol] - x_target).abs().idxmin(), xcol]
slice_df = df[df[xcol] == x_sel]
if zcol:
    z_target = 0.0
    z_sel = slice_df.loc[(slice_df[zcol] - z_target).abs().idxmin(), zcol]
    slice_df = slice_df[slice_df[zcol] == z_sel]
else:
    z_sel = 0.0

# Sort by y and overlay as markers
line_df = slice_df[[ycol, Vcol]].dropna().sort_values(ycol)
plt.plot(line_df[ycol].values, line_df[Vcol].values, linestyle='none', marker='o', markersize=3, label=f'raw CSV (x≈{x_sel:g}, z≈{z_sel:g})')

# Show legend for both curves
plt.legend()
plt.xlabel("y (µm)")
plt.ylabel("V(y) at x=0, z=0 (V)")
plt.title("16-term polynomial: V vs y (x=0, z=0)")
plt.grid(True)
plt.tight_layout()
plt.savefig("V_y_x0_z0_16term.png", dpi=180)
plt.show()
