"""
signal_processing.py
====================
Pre-processing pipeline for raw IMU signals:

1. Butterworth low-pass filter  — removes road vibrations (>5 Hz)
2. Static Detector               — finds when vehicle is stopped (for bias estimation)
3. Bias Estimator                — estimates accelerometer + gyro bias from static periods
4. Madgwick AHRS (simplified)   — estimates pitch, roll, yaw from IMU
5. Phone Alignment              — rotates sensor frame to vehicle frame

WHY FILTERING MATTERS:
A car engine vibrates at 20-100 Hz. Road bumps create spikes up to 10g.
These high-frequency components destroy dead reckoning if not removed first.
The Butterworth filter lets slow "motion" signals through but kills fast "vibration" signals.
"""

import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt, sosfilt
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. Butterworth Low-Pass Filter
# ─────────────────────────────────────────────────────────────────────────────

def butter_lowpass(cutoff_hz: float, fs_hz: float, order: int = 4):
    """
    Design a Butterworth low-pass filter.

    Think of this as a "smoothness dial" — anything faster than cutoff_hz Hz
    gets filtered out. For IMU at 10 Hz, cutoff=4 Hz removes noise above 4 Hz.

    Parameters
    ----------
    cutoff_hz : float — frequency above which noise is removed (e.g., 4.0)
    fs_hz     : float — sampling rate of your data (e.g., 10.0 for IO-VNBD)
    order     : int   — filter sharpness (higher = sharper cutoff, but more delay)
    """
    nyq = fs_hz / 2.0   # Nyquist frequency — max detectable frequency
    normal_cutoff = cutoff_hz / nyq
    normal_cutoff = np.clip(normal_cutoff, 0.01, 0.99)
    sos = butter(order, normal_cutoff, btype="low", analog=False, output="sos")
    return sos


def apply_lowpass(signal: np.ndarray, cutoff_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
    """
    Apply zero-phase Butterworth low-pass filter to a 1D signal.
    Zero-phase = no time delay introduced (good for navigation!).
    """
    sos = butter_lowpass(cutoff_hz, fs_hz, order)
    return sosfiltfilt(sos, signal)


def filter_imu(df: pd.DataFrame, fs_hz: float = 10.0, cutoff_hz: float = 4.5) -> pd.DataFrame:
    """
    Apply low-pass filter to all 6 IMU axes (ax, ay, az, wx, wy, wz).

    At 10 Hz, navigation signals are < 2 Hz (a car turn takes ~3 seconds).
    Road vibrations are 10-100 Hz — already aliased at 10 Hz sampling.
    The filter removes all remnants of vibration energy.
    """
    df = df.copy()
    for col in ["ax", "ay", "az", "wx", "wy", "wz"]:
        if col in df.columns:
            df[f"{col}_filt"] = apply_lowpass(df[col].values, cutoff_hz, fs_hz)
    print(f"[SignalProc] Applied Butterworth LP filter: cutoff={cutoff_hz} Hz")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Static (Zero-Velocity) Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_static_windows(df: pd.DataFrame,
                           acc_threshold: float = 0.3,
                           gyro_threshold: float = 0.05,
                           min_duration_s: float = 1.0,
                           fs_hz: float = 10.0) -> np.ndarray:
    """
    Detect when the vehicle is stationary (stopped at a red light, parked).

    During static periods, the IMU should measure:
    - Accelerometer: [0, 0, g] = [0, 0, 9.81]  (only gravity)
    - Gyroscope: [0, 0, 0]  (no rotation)

    Any deviation from this = sensor bias + noise.
    We use static windows to measure the true sensor bias.

    Returns: boolean array, True = stationary
    """
    ax_col = "ax_filt" if "ax_filt" in df.columns else "ax"
    az_col = "az_filt" if "az_filt" in df.columns else "az"
    wz_col = "wz_filt" if "wz_filt" in df.columns else "wz"

    # Variance in a short window — high variance = moving, low = static
    window = max(int(min_duration_s * fs_hz), 5)

    acc_var = (pd.Series(df[ax_col].values).rolling(window, center=True).std().fillna(1.0))
    gyro_var = (pd.Series(df[wz_col].values).rolling(window, center=True).std().fillna(1.0))

    is_static = (acc_var < acc_threshold) & (gyro_var < gyro_threshold)

    n_static = is_static.sum()
    print(f"[StaticDet] Found {n_static} static samples "
          f"({n_static/len(df)*100:.1f}% of data)")
    return is_static.values


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bias Estimator
# ─────────────────────────────────────────────────────────────────────────────

def estimate_imu_bias(df: pd.DataFrame, is_static: np.ndarray) -> dict:
    """
    Estimate accelerometer and gyroscope biases from static windows.

    During static periods:
    - ax_bias = mean(ax)           (should be 0, but MEMS has offset)
    - ay_bias = mean(ay)           (should be 0)
    - az_bias = mean(az) - 9.81   (should measure gravity = 9.81)
    - wx_bias = mean(wx)           (should be 0)
    - wy_bias = mean(wy)           (should be 0)
    - wz_bias = mean(wz)           (should be 0)

    Returns dict of bias values.
    """
    if is_static.sum() < 50:
        print("[BiasEst] WARNING: Too few static samples! Using zeros as bias.")
        return {col: 0.0 for col in ["ax", "ay", "az", "wx", "wy", "wz"]}

    static_df = df[is_static]

    biases = {}
    for col in ["ax", "ay", "wx", "wy", "wz"]:
        src = f"{col}_filt" if f"{col}_filt" in df.columns else col
        biases[col] = float(static_df[src].mean()) if src in static_df.columns else 0.0

    # az bias relative to gravity
    src_az = "az_filt" if "az_filt" in df.columns else "az"
    biases["az"] = float(static_df[src_az].mean()) - 9.81

    print(f"[BiasEst] Estimated biases: "
          f"ax={biases['ax']:.4f}  ay={biases['ay']:.4f}  az={biases['az']:.4f}  "
          f"wz={biases['wz']:.5f} rad/s")
    return biases


def remove_bias(df: pd.DataFrame, biases: dict) -> pd.DataFrame:
    """Subtract estimated biases from IMU measurements."""
    df = df.copy()
    for col, bias in biases.items():
        src = f"{col}_filt" if f"{col}_filt" in df.columns else col
        dst = f"{col}_corr"
        if src in df.columns:
            df[dst] = df[src] - bias
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Simplified Madgwick AHRS (Attitude & Heading Reference System)
# ─────────────────────────────────────────────────────────────────────────────

class MadgwickAHRS:
    """
    Simplified Madgwick filter for computing roll, pitch, yaw from IMU.

    WHAT IS AHRS?
    Imagine the phone has a tiny virtual "spirit level bubble" + compass.
    - Gyroscope tells us how fast we're rotating (integrate → current angle)
    - Accelerometer tells us which way is "down" (gravity reference)
    - Together they give stable pitch/roll/yaw angles

    WHY NOT JUST INTEGRATE THE GYROSCOPE?
    Gyroscopes drift. After 1 minute, gyro-only heading is off by many degrees.
    The Madgwick filter uses the accelerometer to "anchor" the estimate.

    The output yaw (heading) is what we need for dead reckoning.
    """

    def __init__(self, beta: float = 0.033, freq_hz: float = 10.0):
        """
        beta  : filter gain. Higher = faster correction, but more noisy.
                0.033 is standard for 10 Hz IMU.
        """
        self.beta = beta
        self.dt = 1.0 / freq_hz
        # Quaternion: [qw, qx, qy, qz] — a compact way to represent 3D rotation
        # Initial = no rotation: facing North, level
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, ax: float, ay: float, az: float,
               wx: float, wy: float, wz: float) -> tuple:
        """
        Feed one timestep of IMU data, get back (roll, pitch, yaw) in radians.

        Returns: (roll, pitch, yaw) in radians
        """
        q = self.q
        dt = self.dt

        # Normalize accelerometer (get direction of gravity, ignore magnitude)
        acc_norm = np.sqrt(ax**2 + ay**2 + az**2)
        if acc_norm < 0.1:  # sensor error
            return self._to_euler()
        ax, ay, az = ax/acc_norm, ay/acc_norm, az/acc_norm

        # Gradient descent step (Madgwick's clever math)
        # This computes how much to correct the quaternion using accelerometer
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - ax,
            2*(q[0]*q[1] + q[2]*q[3]) - ay,
            2*(0.5 - q[1]**2 - q[2]**2) - az,
        ])
        J = np.array([
            [-2*q[2],  2*q[3], -2*q[0],  2*q[1]],
            [ 2*q[1],  2*q[0],  2*q[3],  2*q[2]],
            [       0, -4*q[1], -4*q[2],       0],
        ])
        grad = J.T @ f
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad /= grad_norm

        # Gyroscope quaternion derivative
        q_dot = 0.5 * np.array([
            -q[1]*wx - q[2]*wy - q[3]*wz,
             q[0]*wx + q[2]*wz - q[3]*wy,
             q[0]*wy - q[1]*wz + q[3]*wx,
             q[0]*wz + q[1]*wy - q[2]*wx,
        ])

        # Integrate: new quaternion = old + (gyro - beta*correction) * dt
        q = q + (q_dot - self.beta * grad) * dt
        q /= np.linalg.norm(q)  # keep it unit length
        self.q = q

        return self._to_euler()

    def _to_euler(self) -> tuple:
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        q = self.q
        roll  = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
        pitch = np.arcsin (np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1))
        yaw   = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
        return roll, pitch, yaw

    def run_on_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run Madgwick filter on entire DataFrame, adding roll/pitch/yaw columns."""
        ax_col = "ax_corr" if "ax_corr" in df.columns else ("ax_filt" if "ax_filt" in df.columns else "ax")
        ay_col = "ay_corr" if "ay_corr" in df.columns else ("ay_filt" if "ay_filt" in df.columns else "ay")
        az_col = "az_corr" if "az_corr" in df.columns else ("az_filt" if "az_filt" in df.columns else "az")
        wx_col = "wx_corr" if "wx_corr" in df.columns else ("wx_filt" if "wx_filt" in df.columns else "wx")
        wy_col = "wy_corr" if "wy_corr" in df.columns else ("wy_filt" if "wy_filt" in df.columns else "wy")
        wz_col = "wz_corr" if "wz_corr" in df.columns else ("wz_filt" if "wz_filt" in df.columns else "wz")

        rolls, pitches, yaws = [], [], []
        for i in range(len(df)):
            r, p, y = self.update(
                df[ax_col].iloc[i], df[ay_col].iloc[i], df[az_col].iloc[i],
                df[wx_col].iloc[i], df[wy_col].iloc[i], df[wz_col].iloc[i],
            )
            rolls.append(r); pitches.append(p); yaws.append(y)

        df = df.copy()
        df["roll"]    = rolls
        df["pitch"]   = pitches
        df["heading"] = yaws   # yaw = heading = direction vehicle is facing
        print("[Madgwick] Computed roll/pitch/heading for all timesteps")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full Pre-processing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, fs_hz: float = 10.0) -> pd.DataFrame:
    """
    Complete IMU pre-processing pipeline:
    filter → static detect → bias estimate → bias remove → AHRS
    """
    print("\n[PreProcess] Starting IMU pre-processing...")
    df = filter_imu(df, fs_hz=fs_hz, cutoff_hz=min(4.5, fs_hz * 0.45))
    is_static = detect_static_windows(df, fs_hz=fs_hz)
    df["is_static"] = is_static.astype(int)
    biases = estimate_imu_bias(df, is_static)
    df = remove_bias(df, biases)
    ahrs = MadgwickAHRS(beta=0.033, freq_hz=fs_hz)
    df = ahrs.run_on_dataframe(df)
    print("[PreProcess] Done!\n")
    return df, biases


if __name__ == "__main__":
    # Quick test
    from data_loader import generate_synthetic_drive
    df = generate_synthetic_drive(duration_s=300)
    df_proc, biases = preprocess(df)
    print("Estimated bias wz:", biases["wz"])
    print("Computed heading (first 5 values):", df_proc["heading"].values[:5])
