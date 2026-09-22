# Data Pipeline & Signal Conditioning

## End-to-End Pipeline Overview

The data processing pipeline transforms high-frequency, noisy raw sensor telemetry into structured, gravity-stripped feature tensors and Local NED coordinate trajectories.

```
[ Raw Phone IMU (100 Hz) ]          [ GNSS Receiver (1 Hz) ]
          │                                      │
          ▼                                      ▼
[ Sample Rate Synchronization ]       [ WGS-84 to Local NED ]
          │                           (Lat/Lon -> North/East meters)
          ▼                                      │
[ Frame Alignment via DCM ]                      │
(Phone Frame -> Vehicle Body Frame)              │
          │                                      │
          ▼                                      │
[ Gravity Vector Stripping ]                     │
(Subtracts g = 9.80665 m/s² from Z-axis)         │
          │                                      │
          ▼                                      │
[ 4th-Order Butterworth Filter ]                 │
(Low-pass cutoff fc = 15 Hz)                     │
          │                                      │
          ▼                                      │
[ 10-Second Sliding Window ]                     │
(W = 100 samples, 6 features)                    │
          │                                      │
          ▼                                      ▼
   [ BiLSTM Model ]                     [ 8-State EKF ]
```

---

## 1. Geodetic to Local NED Coordinate Conversion

Global Navigation Satellite Systems (GNSS) report positions in **WGS-84 geodetic coordinates**:
- Latitude ($\phi$, in radians)
- Longitude ($\lambda$, in radians)
- Altitude ($h$, in meters)

To perform planar Kalman filtering in metric space, coordinates are mapped to a **Local North-East-Down (NED)** Cartesian tangent plane anchored at the trajectory origin $(\phi_0, \lambda_0, h_0)$:

$$\begin{aligned}
R_M &= \frac{a(1 - e^2)}{(1 - e^2 \sin^2\phi_0)^{3/2}} \quad \text{(Meridian Radius of Curvature)} \\
R_N &= \frac{a}{\sqrt{1 - e^2 \sin^2\phi_0}} \quad \text{(Prime Vertical Radius of Curvature)}
\end{aligned}$$

where the WGS-84 Earth ellipsoid parameters are:
- Semi-major axis: $a = 6,378,137.0\text{ m}$
- First eccentricity squared: $e^2 = 0.00669437999014$

### North and East Displacement Equations:

$$N = (R_M + h_0) \cdot (\phi - \phi_0)$$

$$E = (R_N + h_0) \cos\phi_0 \cdot (\lambda - \lambda_0)$$

---

## 2. Dynamic Direction Cosine Matrix (DCM) & Attitude Alignment

When a driver places a smartphone on a dashboard mount or center console, the phone's internal accelerometer axes $(X_p, Y_p, Z_p)$ are misaligned with the vehicle's forward/lateral driving axes $(X_b, Y_b, Z_b)$.

The pipeline estimates the static/quasi-static tilt angles:
- Pitch angle ($\theta$): rotation about the lateral axis
- Roll angle ($\phi$): rotation about the longitudinal axis

$$\theta = \arctan\left( \frac{-\bar{a}_x}{\sqrt{\bar{a}_y^2 + \bar{a}_z^2}} \right), \quad \phi = \arctan\left( \frac{\bar{a}_y}{\bar{a}_z} \right)$$

The complete Direction Cosine Matrix $C_p^b$ aligning phone coordinates to the vehicle body frame is:

$$C_p^b = \begin{bmatrix}
\cos\theta & \sin\theta \sin\phi & \sin\theta \cos\phi \\
0 & \cos\phi & -\sin\phi \\
-\sin\theta & \cos\theta \sin\phi & \cos\theta \cos\phi
\end{bmatrix}$$

$$\mathbf{a}_{\text{aligned}} = C_p^b \cdot \mathbf{a}_{\text{phone}}, \quad \boldsymbol{\omega}_{\text{aligned}} = C_p^b \cdot \boldsymbol{\omega}_{\text{phone}}$$

---

## 3. Gravity Vector Stripping

In the aligned vehicle frame, Earth's gravity acts along the vertical Down axis ($Z_b$):

$$\mathbf{g}^b = \begin{bmatrix} 0 \\ 0 \\ 9.80665\text{ m/s}^2 \end{bmatrix}$$

The dynamic linear acceleration experienced solely by vehicle motion is:

$$\mathbf{a}_{\text{dynamic}} = \mathbf{a}_{\text{aligned}} - \mathbf{g}^b$$

---

## 4. Digital Butterworth Low-Pass Filtering

Smartphone MEMS accelerometers pick up high-frequency non-kinematic noise from engine mechanical vibrations ($50\text{--}200\text{ Hz}$) and harsh pavement roughness.

We apply a **4th-order zero-phase Butterworth low-pass digital filter**:
- **Sampling Frequency ($f_s$):** $100\text{ Hz}$
- **Cutoff Frequency ($f_c$):** $15\text{ Hz}$
- **Filter Order:** $N = 4$
- **Implementation:** Forward-backward filtering (`scipy.signal.filtfilt`) to ensure **strictly zero phase distortion**.

```python
from scipy.signal import butter, filtfilt

def apply_butterworth_filter(data, cutoff=15.0, fs=100.0, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Zero-phase bidirectional filtering
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data
```

---

## 5. Sliding Window Buffer Construction

The processed 6-channel signals $[\tilde{a}_x, \tilde{a}_y, \tilde{a}_z, \tilde{\omega}_x, \tilde{\omega}_y, \tilde{\omega}_z]$ are partitioned into sliding temporal buffers:
- **Buffer Length ($W$):** 100 samples ($1.0\text{ second}$ of $100\text{ Hz}$ data or $10.0\text{ seconds}$ of $10\text{ Hz}$ decimated data).
- **Buffer Stride:** 1 step per inference epoch.
- **Output Tensor:** Fed directly to the PyTorch BiLSTM velocity inference model.
