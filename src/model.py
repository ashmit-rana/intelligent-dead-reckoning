"""
model.py
========
BiLSTM (Bidirectional Long Short-Term Memory) Speed Estimator.

WHAT DOES THIS MODEL DO?
It looks at a 10-second window of raw IMU data (ax, ay, az, wx, wy, wz)
and predicts: "How fast is this vehicle moving right now?"

WHY BiLSTM?
- LSTM = a neural network that has "memory" — it doesn't just look at the current
  sensor reading but remembers patterns over the last 10 seconds.
- Bi-directional = it reads the window forward AND backward, catching subtle patterns.
- Perfect for time-series sensor data.

WHY IS THIS HARD?
A vehicle at 60 km/h going straight has DIFFERENT IMU patterns than
60 km/h around a curve. The model learns all these patterns from training data.
Once trained, it can estimate speed with no GPS, no OBD-II port — just the phone IMU!

ARCHITECTURE:
  Input: [batch, 100, 6]  (100 timesteps, 6 IMU features)
       ↓
  BiLSTM(64 units) + Dropout(0.2)
       ↓
  BiLSTM(32 units) + Dropout(0.2)
       ↓
  LayerNorm
       ↓
  Linear(32 → 16) + ReLU
       ↓
  Linear(16 → 1)   ← speed prediction in m/s
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

class BiLSTMSpeedEstimator(nn.Module):
    """
    Bidirectional LSTM for vehicle speed estimation from IMU.

    Input features (6): ax, ay, az, wx, wy, wz
    Output: scalar speed (m/s)
    """

    def __init__(self,
                 input_size: int = 6,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM: hidden_size*2 output because bidirectional
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention mechanism: lets the model focus on important timesteps
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size * 2)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.ReLU(),   # speed is always ≥ 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, 6]
        Returns: [batch_size, 1] — predicted speed
        """
        # BiLSTM: output shape [batch, seq_len, hidden*2]
        lstm_out, _ = self.bilstm(x)

        # Attention pooling: weighted average over time dimension
        # Instead of just taking the last timestep, use ALL timesteps
        attn_weights = self.attention(lstm_out)          # [batch, seq, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)   # [batch, hidden*2]

        context = self.norm(self.dropout(context))

        speed = self.regressor(context)
        return speed


def compute_engineered_dataframe(df: pd.DataFrame) -> tuple:
    """
    Unified feature engineering for both training and inference.
    Extracts base IMU signals and computes physically-informed kinematic features:
    - Acceleration magnitude (total and horizontal)
    - Gyroscope magnitude
    - Longitudinal jerk (rate of change of acceleration)
    - Heading angular rate
    - Rolling standard deviation and variance of vertical and forward acceleration
      (road vibration excitation is proportional to vehicle speed)
    """
    df = df.copy()

    # Find base IMU columns (prefer _corr > _filt > raw)
    base_cols = []
    base_map = {}
    for base in ["ax", "ay", "az", "wx", "wy", "wz"]:
        for suffix in ["_corr", "_filt", ""]:
            col = base + suffix
            if col in df.columns:
                base_cols.append(col)
                base_map[base] = col
                break

    ax_c = base_map.get("ax", "ax")
    ay_c = base_map.get("ay", "ay")
    az_c = base_map.get("az", "az")
    wx_c = base_map.get("wx", "wx")
    wy_c = base_map.get("wy", "wy")
    wz_c = base_map.get("wz", "wz")

    # Kinematic magnitudes
    df["feat_acc_mag"]   = np.sqrt(df[ax_c]**2 + df[ay_c]**2 + df[az_c]**2)
    df["feat_acc_horiz"] = np.sqrt(df[ax_c]**2 + df[ay_c]**2)
    df["feat_gyro_mag"]  = np.sqrt(df[wx_c]**2 + df[wy_c]**2 + df[wz_c]**2)
    df["feat_jerk"]      = np.abs(np.gradient(df[ax_c].values))

    if "heading" in df.columns:
        df["feat_heading_rate"] = np.abs(np.gradient(df["heading"].values))
    else:
        df["feat_heading_rate"] = np.abs(df[wz_c].values)

    # Road vibration features: standard deviation and variance (proportional to vehicle speed)
    df["feat_az_std"] = df[az_c].rolling(window=40, min_periods=1).std().fillna(0)
    df["feat_ax_std"] = df[ax_c].rolling(window=40, min_periods=1).std().fillna(0)
    df["feat_az_var"] = df[az_c].rolling(window=40, min_periods=1).var().fillna(0)
    df["feat_ax_var"] = df[ax_c].rolling(window=40, min_periods=1).var().fillna(0)

    engineered = [
        "feat_acc_mag", "feat_acc_horiz", "feat_gyro_mag",
        "feat_jerk", "feat_heading_rate",
        "feat_az_std", "feat_ax_std", "feat_az_var", "feat_ax_var"
    ]
    all_feature_cols = base_cols + [f for f in engineered if f in df.columns]
    return df, all_feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class IMUSpeedDataset(Dataset):
    """
    Sliding window dataset for speed estimation.

    Each sample:
    - X: window of [seq_len] consecutive IMU readings [ax, ay, az, wx, wy, wz, features...]
    - y: speed at the CENTER of that window (ground truth from GPS/wheel encoder)
    """

    def __init__(self,
                 df: pd.DataFrame,
                 seq_len: int = 100,
                 stride: int = 10,
                 scaler: StandardScaler = None,
                 fit_scaler: bool = True):
        self.seq_len = seq_len

        # Compute engineered features
        df_feat, all_features = compute_engineered_dataframe(df)
        self.feature_cols = all_features
        print(f"[Dataset] Using features: {all_features}")

        # Select ground truth speed column
        speed_col = "gt_speed" if "gt_speed" in df.columns else "speed"
        assert speed_col in df.columns, f"No speed column found! Available: {list(df.columns)}"

        # Extract raw arrays — use ALL features including engineered ones
        X_raw = df_feat[all_features].values.astype(np.float32)
        y_raw = df[speed_col].fillna(0).values.astype(np.float32)

        # Normalize features (zero mean, unit variance) — critical for LSTM training
        if scaler is None:
            scaler = StandardScaler()
            if fit_scaler:
                scaler.fit(X_raw)
        self.scaler = scaler
        X_norm = scaler.transform(X_raw)

        # Build windows
        self.X_windows = []
        self.y_labels  = []

        for start in range(0, len(df) - seq_len, stride):
            end = start + seq_len
            window = X_norm[start:end]           # [seq_len, 6]
            # Target: average speed in this window (robust against single-point noise)
            target = np.mean(y_raw[start:end])
            self.X_windows.append(window)
            self.y_labels.append(target)

        self.X_windows = np.array(self.X_windows, dtype=np.float32)
        self.y_labels  = np.array(self.y_labels,  dtype=np.float32)

        print(f"[Dataset] {len(self.X_windows)} windows | "
              f"Speed range: {y_raw.min():.1f}–{y_raw.max():.1f} m/s")

    def __len__(self):
        return len(self.X_windows)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_windows[idx]),
            torch.tensor(self.y_labels[idx]).unsqueeze(0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(df_train: pd.DataFrame,
                df_val: pd.DataFrame,
                seq_len: int = 100,
                stride: int = 5,
                hidden_size: int = 64,
                epochs: int = 50,
                batch_size: int = 64,
                lr: float = 1e-3,
                save_path: str = "models/speed_estimator.pt",
                device: str = None) -> dict:
    """
    Train the BiLSTM speed estimator.

    Returns: dict with training history (losses per epoch)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Train] Using device: {device}")
    print(f"[Train] Epochs={epochs}, Batch={batch_size}, LR={lr}, Hidden={hidden_size}")

    # Build datasets
    train_dataset = IMUSpeedDataset(df_train, seq_len=seq_len, stride=stride, fit_scaler=True)
    val_dataset   = IMUSpeedDataset(df_val,   seq_len=seq_len, stride=stride,
                                   scaler=train_dataset.scaler, fit_scaler=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Build model
    model = BiLSTMSpeedEstimator(
        input_size=len(train_dataset.feature_cols),   # auto-detected (6 raw + 5 engineered = 11)
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.HuberLoss(delta=1.0)  # Huber loss: robust to outliers (potholes!)

    history = {"train_loss": [], "val_loss": [], "val_rmse": []}
    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_losses, all_preds, all_trues = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_losses.append(criterion(pred, y_batch).item())
                all_preds.extend(pred.cpu().numpy().flatten())
                all_trues.extend(y_batch.cpu().numpy().flatten())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        val_rmse   = np.sqrt(np.mean((np.array(all_preds) - np.array(all_trues))**2))
        val_mae    = np.mean(np.abs(np.array(all_preds) - np.array(all_trues)))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val RMSE: {val_rmse:.3f} m/s | "
                  f"Val MAE: {val_mae:.3f} m/s")

        # ── Save best model ───────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_mean": train_dataset.scaler.mean_,
                "scaler_std":  train_dataset.scaler.scale_,
                "feature_cols": train_dataset.feature_cols,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "val_rmse": val_rmse,
                "epoch": epoch,
            }, save_path)
            patience_count = 0
        else:
            patience_count += 1

        # Early stopping: if no improvement for 15 epochs, stop
        if patience_count >= 15:
            print(f"[Train] Early stopping at epoch {epoch}")
            break

    print(f"\n[Train] Best Val RMSE: {min(history['val_rmse']):.4f} m/s")
    print(f"[Train] Model saved to: {save_path}")
    return history, train_dataset.scaler, train_dataset.feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def load_model(save_path: str, device: str = None) -> tuple:
    """Load a saved model checkpoint for inference."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)

    model = BiLSTMSpeedEstimator(
        input_size=len(checkpoint["feature_cols"]),
        hidden_size=checkpoint["hidden_size"],
        num_layers=2,
        dropout=0.0,  # no dropout during inference
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_  = checkpoint["scaler_mean"]
    scaler.scale_ = checkpoint["scaler_std"]

    print(f"[Inference] Loaded model from {save_path} "
          f"(Val RMSE: {checkpoint['val_rmse']:.4f} m/s, "
          f"trained for {checkpoint['epoch']} epochs)")
    return model, scaler, checkpoint["feature_cols"], checkpoint["seq_len"]


def _build_features_for_inference(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Build the same feature matrix used during training.
    Calls compute_engineered_dataframe to ensure 100% train/inference consistency.
    """
    df_feat, _ = compute_engineered_dataframe(df)

    # Select only the columns the model was trained on, in exact order
    available = [c for c in feature_cols if c in df_feat.columns]
    if len(available) < len(feature_cols):
        missing = set(feature_cols) - set(df_feat.columns)
        print(f"[Inference] WARNING: {len(missing)} feature(s) missing: {missing}")
    return df_feat[available].values.astype(np.float32), available



def predict_speed(df: pd.DataFrame,
                  model_path: str = "models/speed_estimator.pt",
                  device: str = None) -> np.ndarray:
    """
    Run speed prediction on an entire DataFrame.
    Returns: np.ndarray of predicted speeds (same length as df)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, feature_cols, seq_len = load_model(model_path, device)

    # Build feature matrix with same engineered features as training
    X_raw, available_cols = _build_features_for_inference(df, feature_cols)
    X_norm = scaler.transform(X_raw)

    predictions = np.zeros(len(df))
    half_window = seq_len // 2

    model.eval()
    with torch.no_grad():
        for i in range(half_window, len(df) - half_window):
            window = X_norm[i - half_window : i + half_window]
            if len(window) < seq_len:
                continue
            x_tensor = torch.tensor(window).unsqueeze(0).to(device)
            pred = model(x_tensor).item()
            predictions[i] = max(0.0, pred)

    # Fill edges
    predictions[:half_window]  = predictions[half_window]
    predictions[-half_window:] = predictions[-half_window-1]

    print(f"[Inference] Predicted speeds — "
          f"Mean: {predictions.mean():.2f} m/s ({predictions.mean()*3.6:.1f} km/h) | "
          f"Max: {predictions.max():.2f} m/s ({predictions.max()*3.6:.1f} km/h)")
    return predictions

