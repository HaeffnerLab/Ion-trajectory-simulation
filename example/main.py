import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
from anharm_analysis.utils import big_plt_font
from anharm_analysis.Trap import COMSOLTrap

big_plt_font() # a specific set of plotting parameters for big font sizes and line width

file = 'Data from comsol/b=2.52a-range(-50,1,50).csv'

# A list of electrode names. In COMSOL simulation a parametric sweep
# should be done with voltages of electrodes ei named as Vei.
#electrodes = ['dc1', 'dc2', 'dc3', 'rf1', 'rf2']

electrodes = ['rf1']

trap = COMSOLTrap(file, electrodes, unit='um', L_ROI=100, sim_unit=1e-6, excitation_prefix='V', sim_prefix='', skiprows=8)

# Find the DC voltage configuration and construct the resulting total potential.
# Higher-order terms can also be passed into it with syntax Cj = <value>.
trap.construct_V_total(C=0, Ey=0, Ez=0, Ex=0, U3=0, U4=0, U2=0, U5=0, U1=1)

# Perform the spherical harmonics expansion with specified order.
ret = trap.expand_spherical_harmonics(order=4)

# --- Materialize Mj coefficients robustly ---
import pandas as _pd, numpy as _np

def _as_1row_df(obj):
    if obj is None:
        return None
    if isinstance(obj, _pd.DataFrame):
        return obj.iloc[[0]] if obj.shape[0] >= 1 else None
    if isinstance(obj, _pd.Series):
        return _pd.DataFrame([obj.values], columns=[str(c) for c in obj.index])
    try:
        arr = _np.asarray(obj)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[0] >= 1:
            return _pd.DataFrame(arr[:1])
    except Exception:
        pass
    return None

# Try multiple likely places the library might store the coefficients
candidates = [
    ret,
    getattr(trap, 'Mj_result', None),
    getattr(trap, 'Mj', None),
    getattr(trap, 'Cj_fit', None),  # often used by plot_Mj
    getattr(trap, 'Cj', None),
    getattr(trap, 'coefficients', None),
]

Mj_df = None
for cand in candidates:
    Mj_df = _as_1row_df(cand)
    if Mj_df is not None:
        break

if Mj_df is None:
    raise RuntimeError("Could not locate coefficients on `trap` to export. Expose `trap.Cj_fit` or `trap.Mj`.")

# Optionally label the first few multipoles with canonical names
_default_names = ['C', 'Ey', 'Ez', 'Ex', 'U3', 'U4', 'U2', 'U5', 'U1']
if all(str(c).isdigit() for c in Mj_df.columns):
    rename_map = {Mj_df.columns[i]: _default_names[i] for i in range(min(len(_default_names), Mj_df.shape[1]))}
    Mj_df.rename(columns=rename_map, inplace=True)

# === OUTPUT: Mj coefficients ===
print("\n=== Full Mj coefficients (no truncation) ===")
with _pd.option_context('display.max_columns', None, 'display.width', 2000):
    print(Mj_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

# 2) Save to files
Mj_df.to_csv("Mj_values.csv", index=False, float_format="%.6e")
try:
    Mj_df.to_excel("Mj_values.xlsx", index=False)
except Exception as e:
    print(f"Note: couldn't write XLSX (install 'openpyxl' to enable). Details: {e}")
Mj_df.to_json("Mj_values.json", orient="records", indent=2)

# 3) Export as a Python vector named C (same column order)
C = Mj_df.iloc[0].tolist()
with open("Mj_vector.py", "w") as f:
    f.write("C = [" + ", ".join(f"{v:.6e}" for v in C) + "]\n")
# Also a plain text version (comma-separated)
_np.savetxt("Mj_vector.txt", C, fmt="%.6e", delimiter=",")

print(f"Saved: Mj_values.csv, Mj_values.json, Mj_vector.py, Mj_vector.txt (N={len(C)})")

trap.plot_V_fit()

trap.plot_Mj(logy=True)

#trap.plot_V_DC()

#trap.plot_potential_contours()

#trap.plot_cutline_fits()

#trap.plot_estimated_frequency_shift()
