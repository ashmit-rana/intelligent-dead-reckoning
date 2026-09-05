"""
evaluation.py
=============
Metrics computation and visualization for the IDR system.

Generates all plots needed for the SIH jury presentation:
1. Trajectory comparison plot (GPS truth vs Raw DR vs EKF vs AI-EKF)
2. Position error over time
3. BiLSTM training loss curve
4. Speed prediction vs ground truth
5. GNSS blackout analysis (before/after comparison)
6. Summary metrics table
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import os


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate Utilities
# ─────────────────────────────────────────────────────────────────────────────

def ned_to_latlon(N: np.ndarray, E: np.ndarray,
                  lat0: float, lon0: float) -> tuple:
    """Convert local NED (meters) back to latitude/longitude."""
    METERS_PER_DEG_LAT = 111320.0
    METERS_PER_DEG_LON = 111320.0 * np.cos(np.radians(lat0))
    lat = lat0 + N / METERS_PER_DEG_LAT
    lon = lon0 + E / METERS_PER_DEG_LON
    return lat, lon


def latlon_to_ned(lat: np.ndarray, lon: np.ndarray,
                  lat0: float, lon0: float) -> tuple:
    """Convert latitude/longitude to local NED (meters)."""
    METERS_PER_DEG_LAT = 111320.0
    METERS_PER_DEG_LON = 111320.0 * np.cos(np.radians(lat0))
    N = (lat - lat0) * METERS_PER_DEG_LAT
    E = (lon - lon0) * METERS_PER_DEG_LON
    return N, E


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_position_error(N_est: np.ndarray, E_est: np.ndarray,
                            N_true: np.ndarray, E_true: np.ndarray) -> np.ndarray:
    """
    Horizontal Position Error (HPE) at each timestep.
    HPE = sqrt((N_est - N_true)² + (E_est - E_true)²)  in meters
    """
    return np.sqrt((N_est - N_true)**2 + (E_est - E_true)**2)


def compute_drift_metrics(N_est: np.ndarray, E_est: np.ndarray,
                          N_true: np.ndarray, E_true: np.ndarray,
                          blackout_mask: np.ndarray = None) -> dict:
    """
    Compute comprehensive drift/accuracy metrics.

    Parameters
    ----------
    blackout_mask : boolean array, True = GNSS denied (only compute drift here)

    Returns dict with RMSE, MAE, max error, drift%, etc.
    """
    error = compute_position_error(N_est, E_est, N_true, E_true)

    metrics = {
        "RMSE_m":      float(np.sqrt(np.mean(error**2))),
        "MAE_m":       float(np.mean(error)),
        "Max_error_m": float(np.max(error)),
        "Final_error_m": float(error[-1]),
    }

    # Total distance traveled (ground truth)
    dN = np.diff(N_true)
    dE = np.diff(E_true)
    total_dist = float(np.sum(np.sqrt(dN**2 + dE**2)))
    metrics["Total_distance_m"] = total_dist

    if blackout_mask is not None and blackout_mask.sum() > 0:
        bo_error = error[blackout_mask]

        # Calculate distance traveled strictly inside contiguous blackout segments
        segments = []
        in_bo = False
        start_i = 0
        for i, bo in enumerate(blackout_mask):
            if bo and not in_bo:
                start_i = i
                in_bo = True
            elif not bo and in_bo:
                segments.append((start_i, i))
                in_bo = False
        if in_bo:
            segments.append((start_i, len(blackout_mask)))

        dist_bo = 0.0
        for s, e in segments:
            if e > s + 1:
                dist_bo += float(np.sum(np.sqrt(np.diff(N_true[s:e])**2 + np.diff(E_true[s:e])**2)))

        metrics["Blackout_RMSE_m"]     = float(np.sqrt(np.mean(bo_error**2)))
        metrics["Blackout_max_error_m"] = float(np.max(bo_error))
        metrics["Blackout_distance_m"]  = dist_bo
        metrics["Drift_percent"] = (metrics["Blackout_max_error_m"] / max(dist_bo, 1.0)) * 100.0

    return metrics


def print_metrics_table(metrics_dict: dict, title: str = "Navigation Metrics"):
    """Print a nicely formatted metrics table for presentation."""
    print("\n" + "="*55)
    print(f"  {title}")
    print("="*55)

    rows = [
        ("RMSE (overall)",          "RMSE_m",              "m"),
        ("MAE (overall)",           "MAE_m",               "m"),
        ("Max Position Error",      "Max_error_m",         "m"),
        ("Final Position Error",    "Final_error_m",       "m"),
        ("Total Distance",          "Total_distance_m",    "m"),
        ("Blackout RMSE",           "Blackout_RMSE_m",     "m"),
        ("Blackout Max Error",      "Blackout_max_error_m","m"),
        ("Blackout Distance",       "Blackout_distance_m", "m"),
        ("Drift Percentage",        "Drift_percent",       "%"),
    ]

    for label, key, unit in rows:
        if key in metrics_dict:
            val = metrics_dict[key]
            color = ""
            if key == "Drift_percent":
                status = "✅ PASS" if val < 10 else "❌ FAIL"
                print(f"  {label:<28} {val:>8.2f} {unit}  {status}")
            else:
                print(f"  {label:<28} {val:>8.2f} {unit}")
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Trajectory Comparison (The MAIN jury plot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectory_comparison(df: pd.DataFrame,
                                dr_N: np.ndarray, dr_E: np.ndarray,
                                ekf_N: np.ndarray, ekf_E: np.ndarray,
                                ai_ekf_N: np.ndarray, ai_ekf_E: np.ndarray,
                                save_path: str = "results/figures/trajectory.png"):
    """
    THE KEY PLOT for jury presentation.
    Shows 4 trajectories:
    - Green: GPS Ground Truth (where the vehicle actually went)
    - Red:   Raw Dead Reckoning (no AI, massive drift)
    - Orange: EKF only (GNSS+INS without AI speed)
    - Blue:  AI-Enhanced EKF (our solution — close to green)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Intelligent Dead Reckoning System — Trajectory Comparison",
                 fontsize=14, fontweight="bold")

    lat0 = df["gt_lat"].iloc[0] if "gt_lat" in df.columns else 28.6315
    lon0 = df["gt_lon"].iloc[0] if "gt_lon" in df.columns else 77.2167

    gt_N = df["gt_N"].values if "gt_N" in df.columns else np.zeros(len(df))
    gt_E = df["gt_E"].values if "gt_E" in df.columns else np.zeros(len(df))

    # Detect blackout windows from DataFrame
    if "gnss_ok" in df.columns:
        blackout_mask_arr = (df["gnss_ok"].values == 0)
    else:
        blackout_mask_arr = np.zeros(len(df), dtype=bool)

    # ── Left plot: Full trajectory ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(gt_E, gt_N, "g-",   lw=2.5, label="GPS Ground Truth",        zorder=5)
    ax.plot(dr_E, dr_N, "r--",  lw=1.5, label="Raw Dead Reckoning (no AI)", alpha=0.7, zorder=3)
    ax.plot(ekf_E, ekf_N, "y-", lw=1.5, label="EKF (no AI speed)",       alpha=0.8, zorder=4)
    ax.plot(ai_ekf_E, ai_ekf_N, "b-", lw=2, label="AI-Enhanced EKF (Ours)", zorder=6)

    # Shade blackout regions on the trajectory (using ground-truth E for x-span)
    if blackout_mask_arr.any():
        in_bo, bo_start = False, 0
        for i, bo in enumerate(blackout_mask_arr):
            if bo and not in_bo:
                bo_start = i; in_bo = True
            elif not bo and in_bo:
                e_lo = min(gt_E[bo_start:i].min(), -1)
                e_hi = max(gt_E[bo_start:i].max(),  1)
                ax.axvspan(gt_E[bo_start], gt_E[i-1], color="gray", alpha=0.12)
                in_bo = False
        if in_bo:
            ax.axvspan(gt_E[bo_start], gt_E[-1], color="gray", alpha=0.12)


    ax.set_xlabel("East (m)", fontsize=11)
    ax.set_ylabel("North (m)", fontsize=11)
    ax.set_title("Full Route Trajectory", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Add start/end markers
    ax.plot(gt_E[0], gt_N[0], "g^", ms=12, zorder=7, label="Start")
    ax.plot(gt_E[-1], gt_N[-1], "gs", ms=10, zorder=7, label="End")
    ax.annotate("START", (gt_E[0], gt_N[0]), textcoords="offset points",
                xytext=(5, 5), fontsize=8, color="green")
    ax.annotate("END", (gt_E[-1], gt_N[-1]), textcoords="offset points",
                xytext=(5, 5), fontsize=8, color="green")

    # ── Right plot: Position Error Over Time ──────────────────────────────────
    ax2 = axes[1]
    t_axis = np.arange(len(df)) * 0.1 / 60   # minutes

    err_dr    = compute_position_error(dr_N,     dr_E,     gt_N, gt_E)
    err_ekf   = compute_position_error(ekf_N,    ekf_E,    gt_N, gt_E)
    err_ai    = compute_position_error(ai_ekf_N, ai_ekf_E, gt_N, gt_E)

    ax2.fill_between(t_axis, 0, err_dr, alpha=0.2, color="red")
    ax2.plot(t_axis, err_dr,  "r--", lw=1.5, label=f"Raw DR (RMSE={np.sqrt(np.mean(err_dr**2)):.1f}m)")
    ax2.plot(t_axis, err_ekf, "y-",  lw=1.5, label=f"EKF (RMSE={np.sqrt(np.mean(err_ekf**2)):.1f}m)")
    ax2.plot(t_axis, err_ai,  "b-",  lw=2,   label=f"AI-EKF (RMSE={np.sqrt(np.mean(err_ai**2)):.1f}m)")

    # Shade GNSS blackout windows — group contiguous spans for efficiency
    if blackout_mask_arr is not None and blackout_mask_arr.any():
        in_bo, bo_start = False, 0
        for i, bo in enumerate(blackout_mask_arr):
            if bo and not in_bo:
                bo_start = i; in_bo = True
            elif not bo and in_bo:
                ax2.axvspan(t_axis[bo_start], t_axis[i-1], color="gray", alpha=0.15)
                in_bo = False
        if in_bo:
            ax2.axvspan(t_axis[bo_start], t_axis[-1], color="gray", alpha=0.15)

    ax2.axhline(y=10, color="red", linestyle=":", lw=1.5, alpha=0.7, label="10% drift ref.")
    # Build legend with a custom proxy for the blackout shading
    handles, labels = ax2.get_legend_handles_labels()
    bo_handle = Line2D([0], [0], color="gray", linewidth=10, alpha=0.4, label="GNSS Blackout")
    ax2.set_xlabel("Time (minutes)", fontsize=11)
    ax2.set_ylabel("Position Error (m)", fontsize=11)
    ax2.set_title("Horizontal Position Error Over Time", fontsize=12)
    ax2.legend(handles=handles + [bo_handle], fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved trajectory comparison → {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: BiLSTM Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict,
                          save_path: str = "results/figures/training.png"):
    """Show the BiLSTM training convergence."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("BiLSTM Speed Estimator — Training Progress", fontsize=13)

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss (Huber)")
    axes[0].plot(epochs, history["val_loss"],   "r-", label="Val Loss (Huber)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_rmse"], "g-", lw=2)
    best_rmse = min(history["val_rmse"])
    best_ep   = history["val_rmse"].index(best_rmse) + 1
    axes[1].axhline(y=best_rmse, color="red", linestyle="--", alpha=0.7)
    axes[1].annotate(f"Best RMSE: {best_rmse:.3f} m/s",
                     xy=(best_ep, best_rmse), xytext=(best_ep+2, best_rmse+0.1),
                     fontsize=9, color="red",
                     arrowprops=dict(arrowstyle="->", color="red"))
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE (m/s)")
    axes[1].set_title("Validation RMSE (↓ is better)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved training curves → {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Speed Prediction vs Ground Truth
# ─────────────────────────────────────────────────────────────────────────────

def plot_speed_prediction(df: pd.DataFrame,
                           speed_pred: np.ndarray,
                           save_path: str = "results/figures/speed.png"):
    """Compare BiLSTM predicted speed vs ground truth."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle("AI Speed Estimator — BiLSTM Prediction vs Ground Truth", fontsize=13)

    t = df["time"].values / 60   # minutes
    gt_col = "gt_speed" if "gt_speed" in df.columns else "speed"
    gt_speed = df[gt_col].fillna(0).values * 3.6    # m/s → km/h
    pred_kmh = speed_pred * 3.6

    axes[0].plot(t, gt_speed,  "g-",  lw=1.5, label="Ground Truth (wheel encoder)", alpha=0.8)
    axes[0].plot(t, pred_kmh, "b--", lw=1.5, label="BiLSTM Prediction (IMU only)", alpha=0.9)
    axes[0].fill_between(t, gt_speed, pred_kmh, alpha=0.15, color="orange", label="Error")
    axes[0].set_ylabel("Speed (km/h)", fontsize=11)
    axes[0].set_title("Speed Comparison")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # GNSS blackout overlay
    if "gnss_ok" in df.columns:
        bo_mask = df["gnss_ok"].values == 0
        axes[0].fill_between(t, 0, gt_speed.max(), where=bo_mask,
                             alpha=0.15, color="red", label="GNSS Blackout")

    error_kmh = pred_kmh - gt_speed
    rmse = np.sqrt(np.mean(error_kmh**2))
    axes[1].plot(t, error_kmh, "r-", lw=1, alpha=0.7)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].fill_between(t, -3.6, 3.6, alpha=0.1, color="green", label="±1 m/s band")
    axes[1].set_xlabel("Time (minutes)", fontsize=11)
    axes[1].set_ylabel("Error (km/h)", fontsize=11)
    axes[1].set_title(f"Prediction Error  (RMSE={rmse:.2f} km/h = {rmse/3.6:.3f} m/s)")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved speed prediction → {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: GNSS Blackout Zoom-In
# ─────────────────────────────────────────────────────────────────────────────

def plot_blackout_analysis(df: pd.DataFrame,
                            dr_N: np.ndarray, dr_E: np.ndarray,
                            ekf_N: np.ndarray, ekf_E: np.ndarray,
                            ai_ekf_N: np.ndarray, ai_ekf_E: np.ndarray,
                            save_path: str = "results/figures/blackout.png"):
    """
    Zoomed-in trajectory during GNSS blackout windows.
    This is the MOST IMPACTFUL plot for jury — shows AI advantage clearly.
    """
    if "gnss_ok" not in df.columns:
        print("[Plot] No gnss_ok column — skipping blackout analysis")
        return

    gt_N = df["gt_N"].values
    gt_E = df["gt_E"].values
    blackout_mask = df["gnss_ok"].values == 0

    # Find contiguous blackout segments
    segments = []
    in_bo = False
    start_i = 0
    for i, bo in enumerate(blackout_mask):
        if bo and not in_bo:
            start_i = i; in_bo = True
        elif not bo and in_bo:
            segments.append((start_i, i))
            in_bo = False
    if in_bo:
        segments.append((start_i, len(blackout_mask)))

    if not segments:
        print("[Plot] No blackout segments found")
        return

    # Select top segments by distance traveled
    segments_with_dist = []
    for s, e in segments:
        d = float(np.sum(np.sqrt(np.diff(gt_N[s:e])**2 + np.diff(gt_E[s:e])**2)))
        segments_with_dist.append(((s, e), d))
    segments_with_dist.sort(key=lambda x: x[1], reverse=True)
    top_segments = [s_e for s_e, d in segments_with_dist[:2]]

    n_segs = len(top_segments)
    fig, axes = plt.subplots(1, n_segs, figsize=(8 * n_segs, 6))
    if n_segs == 1:
        axes = [axes]

    fig.suptitle("GNSS Blackout Zones — Trajectory Zoom", fontsize=13, fontweight="bold")

    for ax, (s, e) in zip(axes, top_segments):
        pad = 20  # timesteps of context before/after blackout
        s_p = max(0, s - pad)
        e_p = min(len(df), e + pad)

        ax.plot(gt_E[s_p:e_p], gt_N[s_p:e_p], "g-",  lw=2.5, label="GPS Truth")
        ax.plot(dr_E[s_p:e_p], dr_N[s_p:e_p], "r--", lw=1.5, label=f"Raw DR")
        ax.plot(ekf_E[s_p:e_p], ekf_N[s_p:e_p], "y-", lw=1.5, label="EKF only")
        ax.plot(ai_ekf_E[s_p:e_p], ai_ekf_N[s_p:e_p], "b-", lw=2, label="AI-EKF (Ours)")

        # Highlight blackout zone path
        ax.plot(gt_E[s:e], gt_N[s:e], color="orange", lw=5, alpha=0.4, label="GNSS Denied Zone")

        # Compute errors at end of blackout
        dr_err  = compute_position_error(dr_N[e-1:e], dr_E[e-1:e], gt_N[e-1:e], gt_E[e-1:e])[0]
        ai_err  = compute_position_error(ai_ekf_N[e-1:e], ai_ekf_E[e-1:e], gt_N[e-1:e], gt_E[e-1:e])[0]
        bo_dist = float(np.sum(np.sqrt(np.diff(gt_N[s:e])**2 + np.diff(gt_E[s:e])**2)))

        ax.set_title(
            f"Blackout Zone (duration={(e-s)*0.1:.0f}s, dist≈{bo_dist:.0f}m)\n"
            f"Raw DR error: {dr_err:.1f}m ({dr_err/max(bo_dist,1)*100:.1f}%)  |  "
            f"AI-EKF error: {ai_err:.1f}m ({ai_err/max(bo_dist,1)*100:.1f}%)",
            fontsize=9
        )
        ax.legend(fontsize=9)
        ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved blackout analysis → {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Summary metrics comparison
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(all_metrics: dict):
    """Print a side-by-side comparison of all methods."""
    print("\n" + "="*75)
    print(f"  PERFORMANCE COMPARISON — SIH Jury Summary")
    print("="*75)
    header = f"{'Metric':<30}" + "".join(f"  {m:<15}" for m in all_metrics)
    print(header)
    print("-"*75)

    key_metrics = [
        ("Overall RMSE (m)",         "RMSE_m"),
        ("Max Position Error (m)",   "Max_error_m"),
        ("Blackout RMSE (m)",         "Blackout_RMSE_m"),
        ("Blackout Max Error (m)",    "Blackout_max_error_m"),
        ("Drift % (< 10% = PASS)",   "Drift_percent"),
    ]

    for label, key in key_metrics:
        row = f"{label:<30}"
        for method, metrics in all_metrics.items():
            val = metrics.get(key, float("nan"))
            if key == "Drift_percent":
                status = "✅" if val < 10 else "❌"
                row += f"  {val:>8.2f}% {status}     "
            else:
                row += f"  {val:>12.2f}m   "
        print(row)

    print("="*75)
    print("Target: Drift < 10% of distance during GNSS blackout")
    print("="*75 + "\n")
