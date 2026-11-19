"""Shared GRF gait pipeline: ingestion -> thresholding -> segmentation -> features.

Used by build_system.py (training), predict.py (command line) and app.py (GUI), so a new
trial is processed exactly the same way the training trials were.

Accepted inputs
  * raw OpenSim .sto files (the 5 %-of-peak threshold is applied here automatically)
  * thresholded trial .csv files (as in the 'threshold data' folders)
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

trapezoid = getattr(np, "trapezoid", None) or np.trapz

# ---- signal processing (same values as the notebook config cell) ----------
N_POINTS       = 101
LOWPASS_HZ     = 20.0
MIN_STANCE_S   = 0.25
MAX_STANCE_S   = 1.50
MERGE_GAP_S    = 0.03
PLATE_ACTIVE_N = 100.0

VERTICAL, AP, ML = "vy", "vx", "vz"
WAVE_COLS = [f"p{i}" for i in range(N_POINTS)]


THRESHOLD_FRAC = 0.05   # vertical GRF below 5 % of the trial's peak is set to 0
SCALAR_FEATURES = [
    "stance_s", "F1_pct", "F1_time_pct", "Fmin_pct", "Fmin_time_pct", "F2_pct",
    "F2_time_pct", "dip_depth_pct", "loading_rate", "vGRF_impulse_norm",
    "AP_brake_pct", "AP_prop_pct", "AP_ratio", "ML_range_pct",
]   # peak_vGRF_N is excluded from modelling: it reflects body weight, not gait pattern


def _as_text(src) -> str:
    """Path, bytes or a file-like upload -> text."""
    if hasattr(src, "read"):
        data = src.read()
        return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    if isinstance(src, bytes):
        return src.decode("utf-8", errors="replace")
    return Path(src).read_text(encoding="utf-8", errors="replace")


def _numeric(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    if "time" not in df.columns:
        raise ValueError(f"{name}: no 'time' column")
    df = df.drop(columns=[c for c in ("source_file", "threshold_N") if c in df.columns])
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["time"]).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError(f"{name}: fewer than 2 usable rows")
    return df.fillna(0.0)


def read_sto(src, name="trial.sto") -> pd.DataFrame:
    """Parse an OpenSim .sto file (header ends at the 'endheader' line)."""
    lines = _as_text(src).splitlines()
    try:
        end = next(i for i, l in enumerate(lines) if l.strip() == "endheader")
    except StopIteration:
        raise ValueError(f"{name}: no 'endheader' line - not a valid .sto file")
    df = pd.read_csv(io.StringIO("\n".join(lines[end + 1:])), sep="\t")
    return _numeric(df, name)


def apply_threshold(df: pd.DataFrame, frac=THRESHOLD_FRAC) -> pd.DataFrame:
    """Zero each plate's vertical GRF below `frac` x the trial's peak vertical GRF.

    Reproduces the 'threshold data' CSVs exactly (checked against Gait_FB_1:
    threshold 31.7158 N = 5 % of 634.3164 N; other channels untouched).
    """
    df = df.copy()
    vcols = [c for c in ("1_ground_force_vy", "2_ground_force_vy") if c in df.columns]
    if not vcols:
        raise ValueError("no *_ground_force_vy column")
    thr = frac * float(df[vcols].max().max())
    for c in vcols:
        df.loc[df[c] < thr, c] = 0.0
    return df


def read_trial(src, name=None) -> pd.DataFrame:
    """Load a trial from a .sto (thresholded here) or an already-thresholded .csv."""
    name = name or Path(getattr(src, "name", str(src))).name
    if name.lower().endswith(".sto"):
        return apply_threshold(read_sto(src, name))
    text = _as_text(src)
    return _numeric(pd.read_csv(io.StringIO(text)), name)


def plate_frame(df: pd.DataFrame, plate: int) -> pd.DataFrame:
    cols = {f"{plate}_ground_force_v{a}": f"v{a}" for a in "xyz"}
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"plate {plate}: missing columns {missing}")
    return df[["time"] + list(cols)].rename(columns=cols).copy()


def active_plates(df: pd.DataFrame) -> list:
    return [p for p in (1, 2)
            if f"{p}_ground_force_{VERTICAL}" in df.columns
            and df[f"{p}_ground_force_{VERTICAL}"].max() > PLATE_ACTIVE_N]


def lowpass(sig, fs, cutoff=LOWPASS_HZ, order=4):
    nyq = fs / 2.0
    if cutoff >= nyq or len(sig) < 4 * order:
        return sig
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, sig)


def find_contacts(vy, time):
    """[(start_idx, end_idx)] for each valid foot contact (contiguous runs of vy > 0)."""
    peak = float(vy.max())
    if peak <= 0:
        return []
    runs, start = [], None
    for i, on in enumerate(vy > 0):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start, i - 1]); start = None
    if start is not None:
        runs.append([start, len(vy) - 1])

    merged = []
    for r in runs:
        if merged and time[r[0]] - time[merged[-1][1]] < MERGE_GAP_S:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))

    return [(a, b) for a, b in merged
            if MIN_STANCE_S <= time[b] - time[a] <= MAX_STANCE_S
            and vy[a:b + 1].max() > 0.4 * peak]


def resample_101(seg):
    return np.interp(np.linspace(0, 100, N_POINTS), np.linspace(0, 100, len(seg)), seg)


def extract_features(vy_n, vx_n, vz_n, stance_s, peak_n) -> dict:
    f1_i = 15 + int(np.argmax(vy_n[15:45]))
    f2_i = 55 + int(np.argmax(vy_n[55:85]))
    mid = vy_n[f1_i:f2_i + 1]
    vmin_i = f1_i + int(np.argmin(mid)) if len(mid) else f1_i
    vmin = vy_n[vmin_i]
    hits = np.flatnonzero(vy_n >= 0.8 * vy_n[f1_i])
    t80 = (hits[0] / 100.0) * stance_s if len(hits) else np.nan
    return {
        "stance_s":          stance_s,
        "peak_vGRF_N":       peak_n,
        "F1_pct":            vy_n[f1_i] * 100,
        "F1_time_pct":       f1_i,
        "Fmin_pct":          vmin * 100,
        "Fmin_time_pct":     vmin_i,
        "F2_pct":            vy_n[f2_i] * 100,
        "F2_time_pct":       f2_i,
        "dip_depth_pct":     (min(vy_n[f1_i], vy_n[f2_i]) - vmin) * 100,
        "loading_rate":      (0.8 * vy_n[f1_i]) / t80 if t80 else np.nan,
        "vGRF_impulse_norm": trapezoid(vy_n, dx=stance_s / 100.0) / stance_s,
        "AP_brake_pct":      vx_n.min() * 100,
        "AP_prop_pct":       vx_n.max() * 100,
        "AP_ratio":          abs(vx_n.min()) / max(vx_n.max(), 1e-9),
        "ML_range_pct":      (vz_n.max() - vz_n.min()) * 100,
    }


def process_trial(src, name=None):
    """Segment one trial (.sto or thresholded .csv; path or uploaded file) into contacts.

    Returns (features_df, waves_df, rejects) where each row is one foot contact and
    waves_df holds the 101-point peak-normalised vertical GRF curve (p0..p100).
    """
    name = name or Path(getattr(src, "name", str(src))).name
    trial = Path(name).stem
    rows, waves, rejects = [], [], []
    try:
        df = read_trial(src, name)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as e:
        return pd.DataFrame(), pd.DataFrame(), [(trial, str(e))]

    dt = np.diff(df["time"].values)
    if not np.all(np.isfinite(dt)) or np.median(dt) <= 0:
        return pd.DataFrame(), pd.DataFrame(), [(trial, "time column is not increasing")]
    fs = float(round(1.0 / np.median(dt)))
    plates = active_plates(df)
    if not plates:
        return pd.DataFrame(), pd.DataFrame(), [(trial, f"no plate exceeds {PLATE_ACTIVE_N:.0f} N")]

    for plate in plates:
        pf = plate_frame(df, plate)
        t, vy_raw = pf["time"].values, pf[VERTICAL].values
        contacts = find_contacts(vy_raw, t)
        if not contacts:
            rejects.append((f"{trial} P{plate}",
                            f"no contact lasting {MIN_STANCE_S}-{MAX_STANCE_S} s"))
            continue
        for n, (a, b) in enumerate(contacts, start=1):
            seg_y = lowpass(vy_raw[a:b + 1], fs)
            seg_x = lowpass(pf[AP].values[a:b + 1], fs)
            seg_z = lowpass(pf[ML].values[a:b + 1], fs)
            peak = float(seg_y.max())
            stance_s = float(t[b] - t[a])
            if peak <= 0:
                continue
            vy_n = resample_101(seg_y) / peak
            vx_n = resample_101(seg_x) / peak
            vz_n = resample_101(seg_z) / peak
            meta = {"trial": trial, "plate": plate, "contact": n}
            rows.append({**meta, "fs_hz": fs, **extract_features(vy_n, vx_n, vz_n, stance_s, peak)})
            waves.append({**meta, **dict(zip(WAVE_COLS, vy_n))})

    return pd.DataFrame(rows), pd.DataFrame(waves), rejects
