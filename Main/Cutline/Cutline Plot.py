# ==== Cutline plots along x, y, z from COMSOL CSV ====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob

# Helper: try to resolve the CSV path from several sensible locations
def resolve_csv_path(default_name):
    """Return an existing CSV path by trying several sensible locations.
    Priority order:
    1) Absolute path as given
    2) CLI arg 1 (if provided)
    3) Env var CUTLINE_CSV
    4) CWD
    5) Script directory
    6) Parent of script directory (project root)
    7) /mnt/data (for uploaded files during notebooks)
    8) Recursive glob under project root and CWD
    9) A glob match in script directory
    """
    # 1) If caller already passed an absolute path that exists, keep it
    if default_name and os.path.isabs(default_name) and os.path.exists(default_name):
        return default_name

    # 2) If CLI arg provided, prefer that
    if len(sys.argv) > 1:
        candidate = sys.argv[1]
        if os.path.exists(candidate):
            return candidate

    # 3) Env var
    env_candidate = os.environ.get('CUTLINE_CSV')
    if env_candidate and os.path.exists(env_candidate):
        return env_candidate

    # 4) Try current working directory
    cwd_candidate = os.path.join(os.getcwd(), default_name)
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    # 5) Try script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sd_candidate = os.path.join(script_dir, default_name)
    if os.path.exists(sd_candidate):
        return sd_candidate

    # 6) Try parent of script directory
    parent_dir = os.path.dirname(script_dir)
    parent_candidate = os.path.join(parent_dir, default_name)
    if os.path.exists(parent_candidate):
        return parent_candidate

    # 7) Try /mnt/data (useful if the CSV was uploaded in a notebook/runtime)
    mnt_candidate = os.path.join('/mnt/data', default_name)
    if os.path.exists(mnt_candidate):
        return mnt_candidate

    # 8) Recursive search under project root and CWD for an exact basename match only
    # Define a simple basename normalizer to compare names ignoring case and trailing parentheses details
    def _base_key(p):
        b = os.path.basename(p).lower().strip()
        # compare up to first parenthesis when present
        prefix = b.split('(')[0]
        return prefix
    target_key = _base_key(default_name)

    # Search roots
    roots = [parent_dir, os.getcwd()]
    for root in roots:
        for p in glob.glob(os.path.join(root, '**', '*.csv'), recursive=True):
            if not os.path.exists(p):
                continue
            # exact basename match first
            if os.path.basename(p).lower() == os.path.basename(default_name).lower():
                return p

    # 9) As a last resort, try to find a similarly named CSV in script_dir
    pattern = os.path.join(script_dir, '*.csv')
    for p in glob.glob(pattern):
        if os.path.basename(p).lower().startswith(os.path.basename(default_name).split('(')[0].lower()):
            return p

    # Nothing found
    raise FileNotFoundError(
        f"Could not locate CSV '{default_name}'.\n"
        f"Tried: CWD='{cwd_candidate}', script_dir='{sd_candidate}', parent='{parent_candidate}', /mnt/data='{mnt_candidate}'.\n"
        f"Tip: run as\n  python {os.path.basename(__file__)} {default_name}")

# --- Load CSV with a best-effort header guess (your file already uses skiprows=8 above)
def _load_csv_best_effort(path):
    # Try with header=None then try with header=0
    try:
        df = pd.read_csv(path, skiprows=8)
    except Exception:
        df = pd.read_csv(path)

    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # allow weird COMSOL headers like "% x" or long potential labels
    def _norm_name(s: str) -> str:
        s = s.strip().lower()
        # drop leading comment markers and extra whitespace
        if s.startswith('%'):
            s = s[1:].strip()
        # collapse multiple spaces
        s = ' '.join(s.split())
        return s

    cols = list(df.columns)
    norm_map = {c: _norm_name(c) for c in cols}

    # Map common aliases (robust to odd COMSOL headers)
    colmap = {}

    # ---- coordinates ----
    # 1) try exact/common aliases
    x_aliases = ['x', 'xcoord', 'x (um)', 'x [um]', 'x [mm]']
    y_aliases = ['y', 'ycoord', 'y (um)', 'y [um]', 'y [mm]']
    z_aliases = ['z', 'zcoord', 'z (um)', 'z [um]', 'z [mm]']

    for k in x_aliases:
        if k in df.columns: colmap['x'] = k; break
    for k in y_aliases:
        if k in df.columns: colmap['y'] = k; break
    for k in z_aliases:
        if k in df.columns: colmap['z'] = k; break

    # 2) fallback: match by normalized prefix (handles "% x", "x   ", etc.)
    if 'x' not in colmap:
        for orig, nm in norm_map.items():
            if nm.startswith('x'):
                colmap['x'] = orig; break
    if 'y' not in colmap:
        for orig, nm in norm_map.items():
            if nm.startswith('y'):
                colmap['y'] = orig; break
    if 'z' not in colmap:
        for orig, nm in norm_map.items():
            if nm.startswith('z'):
                colmap['z'] = orig; break

    # ---- potential ----
    # 1) try common exact names first
    v_aliases = ['v', 'phi', 'potential', 'v_total', 'v (v)', 'electric potential (v)']
    for k in v_aliases:
        if k in df.columns: colmap['V'] = k; break

    # 2) fallback: look for typical COMSOL patterns like
    #    'v (v) @ 1: vrf1=...; ...' or anything containing 'electric potential'
    if 'V' not in colmap:
        for c in df.columns:
            nm = norm_map[c]
            if nm.startswith('v (v)') or 'electric potential' in nm or nm == 'v' or nm == 'phi' or nm == 'potential':
                colmap['V'] = c; break

    missing = {'x','y','z','V'} - set(colmap.keys())
    if missing:
        debug_cols = list(df.columns)
        debug_norm = [norm_map[c] for c in df.columns]
        raise ValueError(
            f"Could not find columns for {missing}.\n"
            f"Original columns: {debug_cols}\n"
            f"Normalized: {debug_norm}\n"
        )

    df = df.rename(columns={colmap['x']:'x', colmap['y']:'y', colmap['z']:'z', colmap['V']:'V'})
    return df[['x','y','z','V']].copy()

csv_path = resolve_csv_path("b=2.52a-range(-10,1,30).csv")
df_raw = _load_csv_best_effort(csv_path)
print(f"[INFO] Loaded CSV: {csv_path}")

# --- Choose center for cutlines
# Strategy: use the coordinate value closest to 0 if present; otherwise take the median grid value.
def _center_from_grid(vals):
    vals_unique = np.unique(np.asarray(vals))
    # pick value closest to zero if zero-ish exists, else the median
    idx = np.argmin(np.abs(vals_unique))
    near0 = vals_unique[idx]
    return near0 if np.abs(near0) <= np.abs(np.median(vals_unique)) * 0.25 or np.isclose(near0, 0) else np.median(vals_unique)

x0 = _center_from_grid(df_raw['x'].values)
y0 = _center_from_grid(df_raw['y'].values)
z0 = _center_from_grid(df_raw['z'].values)

# --- Choose slab thickness automatically from the grid spacing
def _estimate_spacing(vals):
    vals_sorted = np.unique(np.sort(vals))
    diffs = np.diff(vals_sorted)
    diffs = diffs[diffs > 0]
    return np.median(diffs) if diffs.size else 0.0

hx = _estimate_spacing(df_raw['x'].values)
hy = _estimate_spacing(df_raw['y'].values)
hz = _estimate_spacing(df_raw['z'].values)

# slab thickness = 1.5 grid steps (fallback to a small fraction of range)
def _slab(w, default_frac=0.01):
    return 1.5*w if w > 0 else default_frac

slab_y = _slab(hy, 0.01*(df_raw['y'].max()-df_raw['y'].min()))
slab_z = _slab(hz, 0.01*(df_raw['z'].max()-df_raw['z'].min()))
slab_x = _slab(hx, 0.01*(df_raw['x'].max()-df_raw['x'].min()))

# --- Helper: extract a cutline along an axis by taking a thin slab around the other two coords
def extract_cutline(df, axis='x', center=(x0,y0,z0), slabs=(slab_x, slab_y, slab_z)):
    cx, cy, cz = center
    sx, sy, sz = slabs
    if axis == 'x':
        mask = (np.abs(df['y']-cy) <= sy/2) & (np.abs(df['z']-cz) <= sz/2)
        line = df.loc[mask, ['x','V']].copy().sort_values('x')
        line.rename(columns={'x':'coord'}, inplace=True)
        return line, f"x @ y≈{cy:.3g}, z≈{cz:.3g}"
    elif axis == 'y':
        mask = (np.abs(df['x']-cx) <= sx/2) & (np.abs(df['z']-cz) <= sz/2)
        line = df.loc[mask, ['y','V']].copy().sort_values('y')
        line.rename(columns={'y':'coord'}, inplace=True)
        return line, f"y @ x≈{cx:.3g}, z≈{cz:.3g}"
    elif axis == 'z':
        mask = (np.abs(df['x']-cx) <= sx/2) & (np.abs(df['y']-cy) <= sy/2)
        line = df.loc[mask, ['z','V']].copy().sort_values('z')
        line.rename(columns={'z':'coord'}, inplace=True)
        return line, f"z @ x≈{cx:.3g}, y≈{cy:.3g}"
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'.")

cut_x, lbl_x = extract_cutline(df_raw, 'x', (x0,y0,z0), (slab_x, slab_y, slab_z))
cut_y, lbl_y = extract_cutline(df_raw, 'y', (x0,y0,z0), (slab_x, slab_y, slab_z))
cut_z, lbl_z = extract_cutline(df_raw, 'z', (x0,y0,z0), (slab_x, slab_y, slab_z))

# --- Plot (three separate figures for clarity)
def _plot_cutline(cut_df, axis_label, title_suffix):
    if cut_df.empty:
        print(f"[WARN] No points found for {axis_label}-cut at {title_suffix}. "
              f"Try widening the slab thickness.")
        return
    plt.figure()
    plt.plot(cut_df['coord'].values, cut_df['V'].values, lw=2)
    plt.xlabel(f"{axis_label} (same unit as CSV)")   # e.g., μm if your CSV is in μm
    plt.ylabel("Potential V (V)")
    plt.title(f"Cutline along {axis_label} — {title_suffix}")
    plt.grid(True)
    plt.tight_layout()

_plot_cutline(cut_x, 'x', lbl_x)
_plot_cutline(cut_y, 'y', lbl_y)
_plot_cutline(cut_z, 'z', lbl_z)

plt.show()
# ==== End cutline plots ====