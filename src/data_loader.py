"""
data_loader.py
==============
Loads IO-VNBD dataset CSVs or generates synthetic vehicle IMU data
if the real dataset is not yet downloaded.

IO-VNBD Dataset: https://github.com/onyekpeu/IO-VNBD
- Columns can vary by file. This loader auto-detects column names.
- Falls back to synthetic data generation automatically.
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


# ─────────────────────────────────────────────────────────────────────────────
# Column name aliases — IO-VNBD may use different names in different files
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_ALIASES = {
    "time":  ["time", "timestamp", "t", "Time", "Timestamp", "time_s"],
    "ax":    ["ax", "acc_x", "AccX", "accel_x", "a_x", "linear_acceleration_x"],
    "ay":    ["ay", "acc_y", "AccY", "accel_y", "a_y", "linear_acceleration_y"],
    "az":    ["az", "acc_z", "AccZ", "accel_z", "a_z", "linear_acceleration_z"],
    "wx":    ["wx", "gyr_x", "GyroX", "gyro_x", "w_x", "angular_velocity_x"],
    "wy":    ["wy", "gyr_y", "GyroY", "gyro_y", "w_y", "angular_velocity_y"],
    "wz":    ["wz", "gyr_z", "GyroZ", "gyro_z", "w_z", "angular_velocity_z"],
    "lat":   ["lat", "latitude", "Latitude", "GPS_Lat", "gps_lat"],
    "lon":   ["lon", "longitude", "Longitude", "GPS_Lon", "gps_lon"],
    "alt":   ["alt", "altitude", "Altitude", "GPS_Alt"],
    "speed": ["speed", "Speed", "velocity", "v", "wheel_speed", "gps_speed",
              "Speed_ms", "speed_ms", "vehicle_speed"],
}


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names based on aliases."""
    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and canonical not in df.columns:
                rename_map[alias] = canonical
                break
    return df.rename(columns=rename_map)


def load_iovnbd(filepath: str, max_rows: int = None) -> pd.DataFrame:
    """
    Load a single IO-VNBD CSV file and return a normalized DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    max_rows : int, optional
        Limit number of rows for quick testing.

    Returns
    -------
    pd.DataFrame with columns: time, ax, ay, az, wx, wy, wz,
                                lat, lon, alt, speed
    """
    print(f"[DataLoader] Loading: {filepath}")
    df = pd.read_csv(filepath, nrows=max_rows)
    df = _resolve_columns(df)

    required = ["ax", "ay", "az", "wx", "wy", "wz"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required IMU columns after alias resolution: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Create monotonic time axis if missing
    if "time" not in df.columns:
        df["time"] = np.arange(len(df)) * 0.1  # assume 10 Hz

    # Fill optional columns with NaN if missing
    for col in ["lat", "lon", "alt", "speed"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = df["time"] - df["time"].iloc[0]   # start from 0
    df["dt"] = df["time"].diff().fillna(0.1)

    print(f"[DataLoader] Loaded {len(df)} samples, "
          f"duration={df['time'].iloc[-1]:.1f}s, "
          f"GPS available={df['lat'].notna().sum()} rows")
    return df


def load_iovnbd_folder(folder: str, max_files: int = 3, max_rows: int = 50000) -> pd.DataFrame:
    """
    Recursively scan a folder for CSV files and concatenate them.
    Useful for loading the IO-VNBD folder structure.
    """
    csvs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".csv"):
                csvs.append(os.path.join(root, f))

    if not csvs:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    print(f"[DataLoader] Found {len(csvs)} CSV files, loading up to {max_files}...")
    dfs = []
    for path in csvs[:max_files]:
        try:
            dfs.append(load_iovnbd(path, max_rows=max_rows // max_files))
        except Exception as e:
            print(f"  [WARN] Skipping {path}: {e}")

    if not dfs:
        raise RuntimeError("Failed to load any CSV files.")

    combined = pd.concat(dfs, ignore_index=True)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Data Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_drive(
    duration_s: float = 1200.0,   # 20 minutes
    dt: float = 0.1,              # 10 Hz
    noise_level: str = "mems",    # "mems" (phone) or "fog" (high-end)
    seed: int = 42,
    gnss_blackout_windows: list = None,  # list of (start_s, end_s) tuples
) -> pd.DataFrame:
    """
    Generate realistic synthetic vehicle IMU + GPS data.

    Simulates:
    - Straight road segments + turns + roundabouts
    - Traffic stops (red lights)
    - Road vibration & pothole shocks
    - MEMS sensor noise + bias drift
    - GPS outages (configurable)

    Returns a DataFrame with the same schema as IO-VNBD.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s / dt)
    t = np.arange(n) * dt

    # ── True vehicle kinematics ──────────────────────────────────────────────
    # We simulate driving on a "route" with phases: accelerate, cruise, brake, turn
    speed_true = np.zeros(n)     # m/s
    heading_true = np.zeros(n)   # radians (0 = North)

    # Build speed profile: realistic urban/highway driving
    phase_length = n // 12
    phases = [
        ("accelerate", 0, 60/3.6),    # 0 → 60 km/h
        ("cruise",     60/3.6, 60/3.6),
        ("brake",      60/3.6, 20/3.6),
        ("stop",       0, 0),
        ("accelerate", 0, 80/3.6),
        ("cruise",     80/3.6, 80/3.6),
        ("cruise",     80/3.6, 80/3.6),
        ("cruise",     80/3.6, 80/3.6),
        ("brake",      80/3.6, 40/3.6),
        ("cruise",     40/3.6, 40/3.6),
        ("brake",      40/3.6, 0),
        ("stop",       0, 0),
    ]

    idx = 0
    for phase_name, v_start, v_end in phases:
        end_idx = min(idx + phase_length, n)
        speed_true[idx:end_idx] = np.linspace(v_start, v_end, end_idx - idx)
        idx = end_idx

    # Add random micro speed variations (traffic/road roughness)
    speed_true += rng.normal(0, 0.3, n)
    speed_true = np.clip(speed_true, 0, 120/3.6)

    # Build heading profile: turns every ~2 minutes
    turn_events = rng.integers(0, n, size=8)
    turn_angles = rng.uniform(-np.pi/2, np.pi/2, size=8)
    for te, ta in zip(sorted(turn_events), turn_angles):
        turn_len = int(rng.integers(30, 100))  # 3-10 second turn
        end_te = min(te + turn_len, n)
        heading_true[te:end_te] = np.linspace(
            heading_true[te - 1] if te > 0 else 0,
            (heading_true[te - 1] if te > 0 else 0) + ta,
            end_te - te
        )

    # Fill forward any gaps in heading
    for i in range(1, n):
        if heading_true[i] == 0 and i > 0:
            heading_true[i] = heading_true[i - 1]

    # Smooth heading
    from scipy.signal import savgol_filter
    heading_true = savgol_filter(heading_true, 21, 3)

    # ── True position (NED — North, East in meters) ──────────────────────────
    vN_true = speed_true * np.cos(heading_true)
    vE_true = speed_true * np.sin(heading_true)
    N_true = np.cumsum(vN_true) * dt
    E_true = np.cumsum(vE_true) * dt

    # ── True accelerations (body frame) ──────────────────────────────────────
    # Forward acceleration from speed derivative
    a_fwd_true = np.gradient(speed_true, dt)

    # Centripetal acceleration during turns
    heading_rate_true = np.gradient(heading_true, dt)
    a_lat_true = speed_true * heading_rate_true

    # Body-frame accelerations (ax=forward, ay=lateral)
    ax_body_true = a_fwd_true
    ay_body_true = -a_lat_true   # lateral (NHC: should be ~0 on road)
    az_body_true = np.full(n, 9.81)   # gravity (phone vertical)

    # ── MEMS/FOG Noise parameters ─────────────────────────────────────────────
    if noise_level == "mems":
        acc_noise_std  = 0.05    # m/s²
        gyro_noise_std = 0.002   # rad/s
        acc_bias_drift = 0.01    # m/s² drift per step
        gyro_bias_drift = 0.0005 # rad/s drift per step
    else:  # FOG-grade
        acc_noise_std  = 0.001
        gyro_noise_std = 0.00005
        acc_bias_drift = 0.0001
        gyro_bias_drift = 0.00001

    # Slowly drifting biases (random walk)
    acc_bias_x = np.cumsum(rng.normal(0, acc_bias_drift, n))
    acc_bias_y = np.cumsum(rng.normal(0, acc_bias_drift, n))
    gyro_bias_z = np.cumsum(rng.normal(0, gyro_bias_drift, n))

    # ── Speed-proportional road vibration ────────────────────────────────────
    # KEY PHYSICS: road surface induces vibrations at 0.3–2 Hz (from wheel-road
    # interaction at low-speed). Amplitude ∝ speed. These SURVIVE the 4.5 Hz LP
    # filter, giving the BiLSTM a learnable speed-correlated signal.
    #
    # Think of it as: faster car → louder road rumble.
    road_vib_freq1 = 0.5   # Hz — long road waves
    road_vib_freq2 = 1.2   # Hz — road roughness
    road_vib_freq3 = 2.0   # Hz — joint/crack impacts
    phase1 = rng.uniform(0, 2*np.pi)
    phase2 = rng.uniform(0, 2*np.pi)
    phase3 = rng.uniform(0, 2*np.pi)
    road_signal = (
        np.sin(2*np.pi*road_vib_freq1*t + phase1) * 0.012 * speed_true +
        np.sin(2*np.pi*road_vib_freq2*t + phase2) * 0.010 * speed_true +
        np.sin(2*np.pi*road_vib_freq3*t + phase3) * 0.006 * speed_true
    )  # amplitude ∝ speed — key feature for BiLSTM!

    # High-frequency MEMS noise (above 4.5 Hz, will be filtered out)
    hf_noise = 0.08 * rng.normal(0, 1, n)  # filtered by LP

    # Pothole shocks: random impulse events
    potholes = np.zeros(n)
    pothole_times = rng.integers(0, n, size=int(n * 0.003))
    potholes[pothole_times] = rng.uniform(0.3, 1.5, len(pothole_times))

    # ── Measured IMU (noisy) ──────────────────────────────────────────────────
    ax_meas = ax_body_true + acc_bias_x + rng.normal(0, acc_noise_std, n) + road_signal * 0.8 + hf_noise + potholes
    ay_meas = ay_body_true + acc_bias_y + rng.normal(0, acc_noise_std, n) + road_signal * 0.4
    az_meas = az_body_true            + rng.normal(0, acc_noise_std * 2, n) + road_signal + hf_noise * 0.5

    wx_meas = rng.normal(0, gyro_noise_std * 3, n)   # roll rate ≈ 0 on flat road
    wy_meas = rng.normal(0, gyro_noise_std * 3, n)   # pitch rate ≈ 0
    wz_meas = heading_rate_true + gyro_bias_z + rng.normal(0, gyro_noise_std, n)

    # ── GPS (lat/lon) ─────────────────────────────────────────────────────────
    # Origin: New Delhi (Connaught Place)
    lat0, lon0 = 28.6315, 77.2167
    METERS_PER_DEG_LAT = 111320.0
    METERS_PER_DEG_LON = 111320.0 * np.cos(np.radians(lat0))

    # Add GPS noise
    gps_noise_m = 3.0   # ±3 meters (consumer grade)
    lat_meas = lat0 + (N_true + rng.normal(0, gps_noise_m, n)) / METERS_PER_DEG_LAT
    lon_meas = lon0 + (E_true + rng.normal(0, gps_noise_m, n)) / METERS_PER_DEG_LON

    # ── GNSS Blackout windows ─────────────────────────────────────────────────
    if gnss_blackout_windows is None:
        # Default: two blackout windows — one tunnel (60s), one urban canyon (120s)
        gnss_blackout_windows = [
            (200, 260),    # 60-second tunnel
            (600, 720),    # 120-second urban canyon
        ]

    gnss_available = np.ones(n, dtype=bool)
    for start_s, end_s in gnss_blackout_windows:
        start_idx = int(start_s / dt)
        end_idx   = int(end_s   / dt)
        gnss_available[start_idx:end_idx] = False

    lat_gps = np.where(gnss_available, lat_meas, np.nan)
    lon_gps = np.where(gnss_available, lon_meas, np.nan)
    speed_gps = np.where(gnss_available, speed_true + rng.normal(0, 0.3, n), np.nan)

    # ── Pack into DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame({
        "time":       t,
        "dt":         np.full(n, dt),
        "ax":         ax_meas,
        "ay":         ay_meas,
        "az":         az_meas,
        "wx":         wx_meas,
        "wy":         wy_meas,
        "wz":         wz_meas,
        "lat":        lat_gps,
        "lon":        lon_gps,
        "speed":      speed_gps,
        # Ground truth (for evaluation — not available during real inference)
        "gt_speed":   speed_true,
        "gt_heading": heading_true,
        "gt_N":       N_true,
        "gt_E":       E_true,
        "gt_lat":     lat0 + N_true / METERS_PER_DEG_LAT,
        "gt_lon":     lon0 + E_true / METERS_PER_DEG_LON,
        "gnss_ok":    gnss_available.astype(int),
        # True biases (for reference)
        "true_bias_ax": acc_bias_x,
        "true_bias_wz": gyro_bias_z,
    })

    print(f"[SyntheticData] Generated {n} samples @ {1/dt:.0f} Hz | "
          f"Duration={duration_s:.0f}s | "
          f"GNSS blackout: {sum(1-gnss_available)} samples "
          f"({sum(gnss_blackout_windows, ())})")
    return df


def get_data(data_dir: str = None, use_synthetic: bool = False) -> pd.DataFrame:
    """
    High-level data getter.
    - If data_dir is given and CSVs exist → load IO-VNBD
    - Otherwise → generate synthetic data
    """
    if not use_synthetic and data_dir and os.path.isdir(data_dir):
        try:
            df = load_iovnbd_folder(data_dir)
            return df
        except Exception as e:
            print(f"[DataLoader] Could not load IO-VNBD: {e}")
            print("[DataLoader] Falling back to synthetic data...")

    print("[DataLoader] Generating synthetic vehicle data...")
    df = generate_synthetic_drive(
        duration_s=1200,
        gnss_blackout_windows=[(200, 260), (600, 720), (900, 960)]
    )
    return df


if __name__ == "__main__":
    df = get_data(use_synthetic=True)
    print(df.describe())
    print("\nColumns:", list(df.columns))
