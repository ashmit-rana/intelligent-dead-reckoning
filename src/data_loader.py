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


def load_iovnbd_pair(s_path: str, v_path: str = None, max_rows: int = 15000) -> pd.DataFrame:
    """
    Load a synchronized IO-VNBD Smartphone (S) and Vehicle (V) CSV pair.
    Handles latin1 encoding, column stripping, mounting alignment, and ground truth mapping.
    """
    print(f"[DataLoader] Loading smartphone file: {s_path}")
    s = pd.read_csv(s_path, nrows=max_rows, encoding="latin1")
    s.columns = [c.strip() for c in s.columns]

    v = None
    if v_path and os.path.exists(v_path):
        print(f"[DataLoader] Loading vehicle ground truth: {v_path}")
        v = pd.read_csv(v_path, nrows=max_rows, encoding="latin1")
        v.columns = [c.strip() for c in v.columns]

    # Time axis (10 Hz)
    if "TIME SINCE START (ms)" in s.columns:
        t_raw = s["TIME SINCE START (ms)"].values
        time_s = (t_raw - t_raw[0]) / 1000.0
    else:
        time_s = np.arange(len(s)) * 0.1
    dt_arr = np.diff(time_s, prepend=time_s[0] - 0.1)

    # Smartphone raw sensors
    wx = s.get("GYROSCOPE Roll (rad/s)", pd.Series(np.zeros(len(s)))).values
    wy = s.get("GYROSCOPE Pitch (rad/s)", pd.Series(np.zeros(len(s)))).values
    wz = s.get("GYROSCOPE Yaw (rad/s)", pd.Series(np.zeros(len(s)))).values

    # Sensor-to-vehicle mounting projection (calibrated for vehicle yaw rate)
    wz_veh = 0.419215 * wx + 0.962398 * wy + 0.029576 * wz

    # Body frame acceleration (Y is longitudinal forward in vehicle holder, X is lateral)
    if "ACCELEROMETER Y (m/s²)" in s.columns:
        ax = -s["ACCELEROMETER Y (m/s²)"].values
        ay = s["ACCELEROMETER X (m/s²)"].values
        az = s["ACCELEROMETER Z (m/s²)"].values
    else:
        ax = s.get("ax", pd.Series(np.zeros(len(s)))).values
        ay = s.get("ay", pd.Series(np.zeros(len(s)))).values
        az = s.get("az", pd.Series(np.zeros(len(s)))).values

    # Phone GPS
    phone_lat = s.get("GPS LATITUDE (degrees)", pd.Series(np.full(len(s), np.nan))).values
    phone_lon = s.get("GPS LONGITUDE (degrees)", pd.Series(np.full(len(s), np.nan))).values
    if "GPS SPEED (Kmh)" in s.columns:
        raw_spd = s["GPS SPEED (Kmh)"].values
        # IO-VNBD GPS speed values are recorded in m/s directly (max ~19 m/s)
        phone_spd = raw_spd / 3.6 if np.nanmax(raw_spd) > 50 else raw_spd
    else:
        phone_spd = s.get("speed", pd.Series(np.full(len(s), np.nan))).values

    # Vehicle Ground Truth
    if v is not None:
        gt_speed = v.get("Indicated Vehicle Speed (km/hr)", v.get("Velocity (km/hr)", pd.Series(np.zeros(len(s))))).values / 3.6
        gt_lat = v.get("Latitude (degrees)", phone_lat).values
        gt_lon = v.get("Longitude (degrees)", phone_lon).values
        if "Heading (degrees)" in v.columns:
            gt_heading = np.radians(v["Heading (degrees)"].values)
        else:
            gt_heading = np.zeros(len(s))
    else:
        gt_speed = np.copy(phone_spd)
        gt_lat = np.copy(phone_lat)
        gt_lon = np.copy(phone_lon)
        gt_heading = np.zeros(len(s))

    # Compute Local Metric NED coordinates
    valid_idx = np.where(~np.isnan(gt_lat))[0]
    lat0 = gt_lat[valid_idx[0]] if len(valid_idx) > 0 else 52.40166
    lon0 = gt_lon[valid_idx[0]] if len(valid_idx) > 0 else -1.50533

    m_lat = 111320.0
    m_lon = 111320.0 * np.cos(np.radians(lat0))
    gt_N = (gt_lat - lat0) * m_lat
    gt_E = (gt_lon - lon0) * m_lon

    # Inject standard SIH GNSS outage windows (e.g. 20s urban flyovers and tunnels)
    gnss_ok = np.ones(len(s), dtype=int)
    for start_s, end_s in [(300, 320), (500, 520)]:
        st = int(start_s * 10)
        en = min(int(end_s * 10), len(s))
        if st < len(s):
            gnss_ok[st:en] = 0

    meas_lat = np.where(gnss_ok == 1, phone_lat, np.nan)
    meas_lon = np.where(gnss_ok == 1, phone_lon, np.nan)
    meas_spd = np.where(gnss_ok == 1, phone_spd, np.nan)

    df = pd.DataFrame({
        "time": time_s,
        "dt": dt_arr,
        "ax": ax,
        "ay": ay,
        "az": az,
        "wx": wx,
        "wy": wy,
        "wz": wz_veh,
        "lat": meas_lat,
        "lon": meas_lon,
        "speed": meas_spd,
        "gt_speed": gt_speed,
        "gt_lat": gt_lat,
        "gt_lon": gt_lon,
        "gt_heading": gt_heading,
        "gt_N": gt_N,
        "gt_E": gt_E,
        "gnss_ok": gnss_ok,
    })

    print(f"[DataLoader] Successfully loaded {len(df)} samples ({df['time'].iloc[-1]:.1f}s) from real dataset.")
    return df


def load_iovnbd_folder(folder: str, max_rows: int = 15000) -> pd.DataFrame:
    """
    Scan a directory for IO-VNBD files and load them.
    Pairs smartphone files (S-*.csv) with vehicle ground truth files (V-*.csv).
    """
    s_files = []
    v_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".csv"):
                full_path = os.path.join(root, f)
                if f.startswith("S-") or "S (" in full_path:
                    s_files.append(full_path)
                elif f.startswith("V-") or "V (" in full_path:
                    v_files.append(full_path)

    if not s_files:
        # Fallback to any CSV
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".csv"):
                    return load_iovnbd_pair(os.path.join(root, f), max_rows=max_rows)
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    s_path = sorted(s_files)[0]
    # Find matching V file if possible (e.g. S-S1.csv matches V-S1.csv)
    v_path = None
    s_stem = os.path.basename(s_path).replace("S-", "").replace(".csv", "")
    for vf in v_files:
        if s_stem in os.path.basename(vf):
            v_path = vf
            break
    if v_path is None and v_files:
        v_path = sorted(v_files)[0]

    return load_iovnbd_pair(s_path, v_path, max_rows=max_rows)


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
