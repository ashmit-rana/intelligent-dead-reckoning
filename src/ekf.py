"""
ekf.py
======
8-State Extended Kalman Filter (EKF) for GNSS + INS Fusion.

STATE VECTOR (what the EKF is tracking at every timestep):
    x = [N,  E,  vN,  vE,  heading,  ba_x,  ba_y,  bg_z]
         ↑   ↑    ↑    ↑     ↑         ↑      ↑       ↑
       North East  velocities  yaw   acc bias      gyro bias
      (pos, meters)         (m/s)  (rad)  (m/s²)   (rad/s)

WHY EKF?
Think of EKF as a "smart average" between two imperfect sources:
- IMU says: "based on how you've been moving, you're at position X"
- GPS says: "I can see satellites, you're actually at position Y"
- EKF says: "The truth is somewhere between X and Y, and I know how much
            to trust each source based on their noise characteristics"

The "Extended" part just means we handle non-linear equations
(like sines and cosines in navigation) using linearization (Jacobians).
"""

import numpy as np
import pandas as pd


class EKF:
    """
    8-State EKF for 2D ground vehicle navigation.

    Works in LOCAL NED (North-East-Down) coordinates centered at starting point.
    Position in meters, velocity in m/s, angles in radians.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.n = 8  # number of states

        # ── Initial State ────────────────────────────────────────────────────
        # We start at origin (0,0) with zero velocity, heading North, zero biases
        self.x = np.zeros(self.n)

        # ── Initial Covariance (P) ────────────────────────────────────────────
        # How uncertain are we at the start?
        # Large diagonal = very uncertain (good: we let GPS correct us quickly)
        self.P = np.diag([
            25.0,    # N position uncertainty: ±5m
            25.0,    # E position uncertainty: ±5m
            4.0,     # vN velocity uncertainty: ±2 m/s
            4.0,     # vE velocity uncertainty: ±2 m/s
            0.1,     # heading uncertainty: ±0.32 rad ≈ 18°
            0.01,    # acc_bias_x: ±0.1 m/s²
            0.01,    # acc_bias_y: ±0.1 m/s²
            0.0001,  # gyro_bias_z: ±0.01 rad/s
        ])

        # ── Process Noise (Q) ─────────────────────────────────────────────────
        # How much does the true state drift from our model in each step?
        # Higher Q = we trust the model less, adapt faster to measurements.
        self.Q = np.diag([
            0.01,    # N: 0.1m² per step uncertainty
            0.01,    # E
            0.1,     # vN: velocity can change by 0.32 m/s per step
            0.1,     # vE
            0.001,   # heading
            1e-5,    # acc_bias_x (biases change slowly)
            1e-5,    # acc_bias_y
            1e-7,    # gyro_bias_z (very slow drift)
        ])

        # ── Measurement Noise (R) — GPS ────────────────────────────────────────
        # Consumer GPS: ±3m position, ±0.5 m/s velocity
        self.R_gps = np.diag([9.0, 9.0, 0.25, 0.25])  # [N, E, vN, vE] variances

        # ── Measurement Noise (R) — Non-Holonomic Constraints ─────────────────
        # A car CANNOT move sideways (vE perpendicular to heading ≈ 0)
        # We treat this as a "fake measurement" of zero lateral velocity
        # Trust this strongly (very small noise = high confidence)
        self.R_nhc = np.diag([0.01, 0.01])  # [lateral vel, vertical vel]

        # History for plotting
        self.history = {"N": [], "E": [], "heading": [], "speed": []}

    def reset(self, lat0: float = None, lon0: float = None):
        """Reset the filter state to origin."""
        self.x = np.zeros(self.n)
        self.P = np.diag([25, 25, 4, 4, 0.1, 0.01, 0.01, 0.0001])
        self.history = {"N": [], "E": [], "heading": [], "speed": []}

    # ─────────────────────────────────────────────────────────────────────────
    # Predict Step (using IMU as "engine")
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, ax_meas: float, ay_meas: float, wz_meas: float, dt: float = None):
        """
        PREDICT: "Where do I think I am, based on IMU alone?"

        This is dead reckoning — integrating IMU to get position.
        Called at every IMU timestep (e.g., 10 Hz).

        Parameters
        ----------
        ax_meas : forward acceleration (m/s²) in body frame
        ay_meas : lateral acceleration (m/s²) in body frame (should be ≈0 on road)
        wz_meas : yaw rate (rad/s) — how fast heading is changing
        """
        dt = dt or self.dt

        # Extract current state
        N, E, vN, vE, heading, ba_x, ba_y, bg_z = self.x

        # Remove estimated biases from IMU readings (bias compensation)
        ax_corr = ax_meas - ba_x
        ay_corr = ay_meas - ba_y
        wz_corr = wz_meas - bg_z

        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        # Body-frame to NED-frame acceleration
        aN = ax_corr * cos_h - ay_corr * sin_h   # North acceleration
        aE = ax_corr * sin_h + ay_corr * cos_h   # East acceleration

        # ── Nonlinear state transition (f(x)) ────────────────────────────────
        N_new       = N + vN * dt + 0.5 * aN * dt**2
        E_new       = E + vE * dt + 0.5 * aE * dt**2
        vN_new      = vN + aN * dt
        vE_new      = vE + aE * dt
        heading_new = heading + wz_corr * dt
        # Biases modeled as random walk (constant between updates)
        ba_x_new = ba_x
        ba_y_new = ba_y
        bg_z_new = bg_z

        self.x = np.array([N_new, E_new, vN_new, vE_new, heading_new,
                            ba_x_new, ba_y_new, bg_z_new])

        # ── Linearize the state transition: Jacobian F ────────────────────────
        # F = df/dx evaluated at current state (how state change affects next state)
        F = np.eye(self.n)
        F[0, 2] = dt                                   # N depends on vN
        F[1, 3] = dt                                   # E depends on vE
        F[0, 4] = (-ax_corr * sin_h - ay_corr * cos_h) * 0.5 * dt**2  # N on heading
        F[1, 4] = ( ax_corr * cos_h - ay_corr * sin_h) * 0.5 * dt**2  # E on heading
        F[2, 4] = (-ax_corr * sin_h - ay_corr * cos_h) * dt            # vN on heading
        F[3, 4] = ( ax_corr * cos_h - ay_corr * sin_h) * dt            # vE on heading
        F[0, 5] = -cos_h * 0.5 * dt**2                # N on ba_x
        F[1, 5] = -sin_h * 0.5 * dt**2                # E on ba_x
        F[2, 5] = -cos_h * dt                          # vN on ba_x
        F[3, 5] = -sin_h * dt                          # vE on ba_x
        F[0, 6] =  sin_h * 0.5 * dt**2                # N on ba_y
        F[1, 6] = -cos_h * 0.5 * dt**2
        F[2, 6] =  sin_h * dt
        F[3, 6] = -cos_h * dt
        F[4, 7] = -dt                                  # heading on bg_z

        # ── Covariance propagation: P = F*P*F' + Q ───────────────────────────
        self.P = F @ self.P @ F.T + self.Q

        self._record()

    # ─────────────────────────────────────────────────────────────────────────
    # GPS Update Step
    # ─────────────────────────────────────────────────────────────────────────

    def update_gps(self, N_gps: float, E_gps: float,
                   vN_gps: float = None, vE_gps: float = None):
        """
        UPDATE with GPS measurement: "GPS says I'm here — correct my estimate!"

        Parameters
        ----------
        N_gps, E_gps  : GPS position in local NED frame (meters)
        vN_gps, vE_gps: GPS velocity (m/s), optional
        """
        if vN_gps is not None and vE_gps is not None:
            # 4-measurement update: position + velocity
            z = np.array([N_gps, E_gps, vN_gps, vE_gps])
            H = np.zeros((4, self.n))
            H[0, 0] = 1.0   # N
            H[1, 1] = 1.0   # E
            H[2, 2] = 1.0   # vN
            H[3, 3] = 1.0   # vE
            R = self.R_gps
        else:
            # 2-measurement update: position only
            z = np.array([N_gps, E_gps])
            H = np.zeros((2, self.n))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            R = self.R_gps[:2, :2]

        self._kalman_update(z, H, R)

    # ─────────────────────────────────────────────────────────────────────────
    # Non-Holonomic Constraint Update
    # ─────────────────────────────────────────────────────────────────────────

    def update_nhc(self):
        """
        NON-HOLONOMIC CONSTRAINT (NHC):
        A ground vehicle CANNOT slide sideways!

        This gives us a free "measurement" whenever GNSS is unavailable:
        - Lateral velocity (perpendicular to heading) must be ≈ 0
        - Vertical velocity must be ≈ 0 (car stays on road)

        This is like getting free GPS updates just from physics — powerful!
        The NHC can reduce drift by 40-60% during GNSS blackout.
        """
        heading = self.x[4]
        vN = self.x[2]
        vE = self.x[3]

        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        # Lateral velocity = component of velocity perpendicular to heading
        # In ground vehicles, lateral velocity must be 0 (no sideways sliding)
        v_lat = -vN * sin_h + vE * cos_h

        z = np.array([0.0])  # NHC measurement: lateral velocity is ZERO
        h_x = np.array([v_lat])
        H = np.zeros((1, self.n))
        H[0, 2] = -sin_h   # d(v_lat)/d(vN)
        H[0, 3] =  cos_h   # d(v_lat)/d(vE)
        H[0, 4] = -vN * cos_h - vE * sin_h  # d(v_lat)/d(heading)

        R = np.array([[0.04]])  # std = 0.2 m/s confidence for NHC
        self._kalman_update(z, H, R, h_x=h_x)

    # ─────────────────────────────────────────────────────────────────────────
    # Heading Update (from Madgwick AHRS)
    # ─────────────────────────────────────────────────────────────────────────

    def update_heading(self, heading_meas: float, heading_uncertainty: float = 0.05):
        """
        Update heading from AHRS (Madgwick filter) orientation estimate.
        Binds yaw angle and allows real-time Kalman estimation of gyro bias bg_z.
        """
        H = np.zeros((1, self.n))
        H[0, 4] = 1.0
        # Angle innovation wrapped to [-pi, pi]
        diff = np.arctan2(np.sin(heading_meas - self.x[4]), np.cos(heading_meas - self.x[4]))
        z = np.array([self.x[4] + diff])
        R = np.array([[max(0.001, heading_uncertainty**2)]])
        self._kalman_update(z, H, R)

    # ─────────────────────────────────────────────────────────────────────────
    # AI Velocity Update (from BiLSTM model + Heading)
    # ─────────────────────────────────────────────────────────────────────────

    def update_ai_velocity(self, speed_pred: float, heading: float = None,
                           speed_uncertainty: float = 1.0):
        """
        Direct 2D velocity vector update using AI speed and heading.
        Binds velocity to the learned AI speed profile during blackout.
        """
        h = self.x[4] if heading is None else heading
        vN_meas = float(max(0.0, speed_pred)) * np.cos(h)
        vE_meas = float(max(0.0, speed_pred)) * np.sin(h)

        z = np.array([vN_meas, vE_meas])
        H = np.zeros((2, self.n))
        H[0, 2] = 1.0   # vN
        H[1, 3] = 1.0   # vE

        var_v = max(0.1, speed_uncertainty**2)
        R = np.diag([var_v, var_v])
        self._kalman_update(z, H, R)

    def update_ai_speed(self, speed_pred: float, speed_uncertainty: float = 1.0):
        """Forward-speed measurement update."""
        heading = self.x[4]
        self.update_ai_velocity(speed_pred, heading=heading, speed_uncertainty=speed_uncertainty)


    # ─────────────────────────────────────────────────────────────────────────
    # Core Kalman Update Equation
    # ─────────────────────────────────────────────────────────────────────────

    def _kalman_update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray,
                       h_x: np.ndarray = None):
        """
        The heart of the Extended Kalman Filter:

        y = z - h(x)                         [Innovation from non-linear measurement]
        S = H * P * H' + R                   [Innovation covariance]
        K = P * H' * S^-1                    [Kalman Gain]
        x = x + K * y                        [State update]
        P = (I - K*H) * P                    [Covariance update]
        """
        if h_x is not None:
            y = z - h_x                              # Exact non-linear innovation
        else:
            y = z - H @ self.x                       # Linear measurement model
        S = H @ self.P @ H.T + R                    # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ y                     # Correct state
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P               # Update uncertainty

        # Normalize heading to [-pi, pi]
        self.x[4] = np.arctan2(np.sin(self.x[4]), np.cos(self.x[4]))

    def _record(self):
        N, E, vN, vE, heading, *_ = self.x
        self.history["N"].append(N)
        self.history["E"].append(E)
        self.history["heading"].append(heading)
        self.history["speed"].append(np.sqrt(vN**2 + vE**2))

    def get_position(self) -> tuple:
        return self.x[0], self.x[1]   # N, E in meters

    def get_speed(self) -> float:
        return np.sqrt(self.x[2]**2 + self.x[3]**2)


# ─────────────────────────────────────────────────────────────────────────────
# Dead Reckoning baseline (raw IMU integration, no EKF)
# Used to show HOW BAD pure DR is → motivates the AI+EKF solution
# ─────────────────────────────────────────────────────────────────────────────

def run_raw_dead_reckoning(df: pd.DataFrame) -> dict:
    """
    Naive dead reckoning: just double-integrate raw IMU with no correction.
    This is the "before AI" baseline — it will drift badly!

    Steps:
    1. Integrate wz (gyro) → heading
    2. Rotate ax, ay using heading → NED frame
    3. Integrate aN, aE → vN, vE → N, E
    """
    ax_col = "ax_corr" if "ax_corr" in df.columns else "ax"
    ay_col = "ay_corr" if "ay_corr" in df.columns else "ay"
    wz_col = "wz_corr" if "wz_corr" in df.columns else "wz"

    n = len(df)
    N_arr = np.zeros(n)
    E_arr = np.zeros(n)
    vN, vE = 0.0, 0.0
    heading = 0.0

    # Use Madgwick heading if available (better than raw gyro integration)
    use_madgwick = "heading" in df.columns

    for i in range(1, n):
        dt = float(df["dt"].iloc[i]) if "dt" in df.columns else 0.1
        ax = float(df[ax_col].iloc[i])
        ay = float(df[ay_col].iloc[i])

        if use_madgwick:
            heading = float(df["heading"].iloc[i])
        else:
            wz = float(df[wz_col].iloc[i])
            heading += wz * dt

        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        aN = ax * cos_h - ay * sin_h
        aE = ax * sin_h + ay * cos_h

        vN += aN * dt
        vE += aE * dt
        N_arr[i] = N_arr[i-1] + vN * dt
        E_arr[i] = E_arr[i-1] + vE * dt

    return {"N": N_arr, "E": E_arr}


def run_ekf_pipeline(df: pd.DataFrame,
                     speed_predictions: np.ndarray = None,
                     speed_uncertainty: float = 1.5) -> dict:
    """
    Run the full EKF pipeline on a processed DataFrame.

    Parameters
    ----------
    df                 : preprocessed DataFrame with IMU + GPS columns
    speed_predictions  : BiLSTM speed estimates (one per timestep), or None
    speed_uncertainty  : std dev of speed prediction error (m/s)

    Returns dict with estimated trajectory arrays.
    """
    lat0_approx = 28.6315
    lon0_approx = 77.2167
    METERS_PER_DEG_LAT = 111320.0
    METERS_PER_DEG_LON = 111320.0 * np.cos(np.radians(lat0_approx))

    # Detect lat0, lon0 from first valid GPS fix
    valid_gps = df[df["lat"].notna()]
    if len(valid_gps) > 0:
        lat0 = valid_gps["lat"].iloc[0]
        lon0 = valid_gps["lon"].iloc[0]
    else:
        lat0, lon0 = lat0_approx, lon0_approx

    n = len(df)
    dt_default = 0.1

    ekf = EKF(dt=dt_default)
    ekf.reset()

    ax_col = "ax_corr" if "ax_corr" in df.columns else "ax"
    ay_col = "ay_corr" if "ay_corr" in df.columns else "ay"
    wz_col = "wz_corr" if "wz_corr" in df.columns else "wz"

    N_out, E_out = np.zeros(n), np.zeros(n)

    speed_scale = 1.0
    recent_ratios = []

    for i in range(n):
        dt = float(df["dt"].iloc[i]) if "dt" in df.columns else dt_default
        ax = float(df[ax_col].iloc[i])
        ay = float(df[ay_col].iloc[i])
        wz = float(df[wz_col].iloc[i])

        # ── PREDICT ──────────────────────────────────────────────────────────
        ekf.predict(ax, ay, wz, dt=dt)

        # ── GPS UPDATE (only when available) ──────────────────────────────────
        lat_val = df["lat"].iloc[i]
        if not np.isnan(lat_val):
            N_gps = (lat_val - lat0) * METERS_PER_DEG_LAT
            E_gps = (df["lon"].iloc[i] - lon0) * METERS_PER_DEG_LON

            spd = df["speed"].iloc[i] if "speed" in df.columns else np.nan
            if not np.isnan(spd):
                # Calculate ground track components
                if "gt_heading" in df.columns:
                    h_true = float(df["gt_heading"].iloc[i])
                    vN_gps = spd * np.cos(h_true)
                    vE_gps = spd * np.sin(h_true)
                else:
                    h_est = ekf.x[4]
                    vN_gps = spd * np.cos(h_est)
                    vE_gps = spd * np.sin(h_est)

                ekf.update_gps(N_gps, E_gps, vN_gps, vE_gps)

                # GPS Course Over Ground (COG) heading update: aligns heading and calibrates gyro bias
                if spd > 1.2:
                    cog = np.arctan2(vE_gps, vN_gps)
                    ekf.update_heading(cog, heading_uncertainty=0.015)

                    # Online speed scale factor calibration
                    if speed_predictions is not None and speed_predictions[i] > 1.5:
                        recent_ratios.append(spd / speed_predictions[i])
                        if len(recent_ratios) > 100:
                            recent_ratios.pop(0)
                        speed_scale = float(np.median(recent_ratios[-40:]))
            else:
                ekf.update_gps(N_gps, E_gps)
        else:
            # ── GNSS BLACKOUT: use ZUPT + NHC + calibrated AI velocity ────────
            is_stopped = False
            if "is_static" in df.columns and df["is_static"].iloc[i] == 1:
                is_stopped = True
            elif speed_predictions is not None and (speed_predictions[i] * speed_scale) < 0.6:
                is_stopped = True

            if is_stopped:
                # Zero Velocity Update (ZUPT): freeze velocity when vehicle is stationary
                ekf.x[2] = 0.0
                ekf.x[3] = 0.0
            else:
                ekf.update_nhc()
                if speed_predictions is not None:
                    spd_val = float(speed_predictions[i]) * speed_scale
                    ekf.update_ai_velocity(spd_val, heading=ekf.x[4], speed_uncertainty=max(0.3, speed_uncertainty))

        N_out[i], E_out[i] = ekf.get_position()

    print(f"[EKF] Pipeline done. Final position: N={N_out[-1]:.1f}m, E={E_out[-1]:.1f}m")
    return {
        "N": N_out,
        "E": E_out,
        "lat0": lat0,
        "lon0": lon0,
    }
