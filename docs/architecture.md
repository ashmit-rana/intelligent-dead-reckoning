# System Architecture & State Estimation Mathematics

## Architecture Diagram

The system operates across three tightly synchronized layers: **Sensor Ingestion & Conditioning**, **Deep Neural Velocity Regression**, and **8-State Extended Kalman Filter (EKF) State Estimation**.

```
+─────────────────────────────────────────────────────────────────────────────────────────+
|                                    SYSTEM ARCHITECTURE                                  |
+─────────────────────────────────────────────────────────────────────────────────────────+

  [ Phone MEMS IMU ]          [ GNSS / NavIC Receiver ]          [ Vehicle CAN-Bus ]
   (100 Hz Accel & Gyro)           (1 Hz Fix & NMEA)             (Validation Ground Truth)
             │                             │                                 │
             ▼                             ▼                                 │
  ┌───────────────────────────────────────────────┐                          │
  │            SIGNAL PREPROCESSING               │                          │
  │  • Coordinate Frame Alignment (DCM Matrix)    │                          │
  │  • Gravity Stripping (g = 9.80665 m/s²)       │                          │
  │  • 4th-Order Butterworth Low-Pass (fc = 15Hz) │                          │
  │  • 1-Second Sliding Buffer (W = 100 samples)  │                          │
  └───────────────────────┬───────────────────────┘                          │
                          │                                                  │
            ┌─────────────┴─────────────┐                                    │
            ▼                           ▼                                    │
  ┌───────────────────┐       ┌───────────────────┐                          │
  │   RAW GYROSCOPE   │       │   BiLSTM NEURAL   │                          │
  │  Angular Rate (ω) │       │   SPEED ENGINE    │                          │
  │  Yaw Rate (rad/s) │       │  Predicted Speed  │                          │
  └─────────┬─────────┘       │    v_ai (m/s)     │                          │
            │                 └─────────┬─────────┘                          │
            │                           │                                    │
            ▼                           ▼                                    │
  ┌───────────────────────────────────────────────────────────────────────┐  │
  │                  8-STATE EXTENDED KALMAN FILTER (EKF)                 │  │
  │                                                                       │  │
  │  State: x = [N, E, vN, vE, ψ, ba_x, ba_y, bg_z]ᵀ                      │  │
  │                                                                       │  │
  │  [Prediction Step]                                                    │  │
  │  x̂ₖ|ₖ₋₁ = f(x̂ₖ₋₁, uₖ),   Pₖ|ₖ₋₁ = Fₖ Pₖ₋₁ Fₖᵀ + Q                        │  │
  │                                                                       │  │
  │  [Correction Step]                                                    │  │
  │  • Satellite Visible  ──► Update with GNSS Pos (N, E)                 │  │
  │  • Tunnel Blackout    ──► Update with v_ai + NHC (vy ≈ 0) + ZUPT      │  │
  │  • Innovation Update  ──► Kₖ = Pₖ|ₖ₋₁ Hᵀ (H Pₖ|ₖ₋₁ Hᵀ + R)⁻¹            │  │
  │                           x̂ₖ|ₖ = x̂ₖ|ₖ₋₁ + Kₖ (zₖ - h(x̂ₖ|ₖ₋₁))            │  │
  │                           Pₖ|ₖ = (I - Kₖ H) Pₖ|ₖ₋₁                      │  │
  └───────────────────────────────────┬───────────────────────────────────┘  │
                                      │                                      │
                                      ▼                                      ▼
                       ┌─────────────────────────────┐        ┌───────────────────────┐
                       │  REAL-TIME NAVIGATION HUD   │◄───────│ BENCHMARK COMPARATOR  │
                       │  • 10 Hz Trajectory (N, E)  │        │ • Raw DR vs EKF vs AI │
                       │  • Heading, Velocity, Drift │        │ • Metrics & Drift %   │
                       └─────────────────────────────┘        └───────────────────────┘
```

---

## Coordinate Frames & Reference Systems

The navigation engine maps between two distinct frames of reference:

1. **Body Frame ($B$):** Attached to the vehicle chassis.
   - $X_b$: Forward (longitudinal driving direction)
   - $Y_b$: Right (lateral axis)
   - $Z_b$: Down (perpendicular to road surface)
2. **Local Navigation Frame ($N$ - Local NED):** Cartesian tangent plane centered at the initial vehicle GNSS lock coordinate ($\phi_0, \lambda_0, h_0$).
   - $N$: North displacement in meters
   - $E$: East displacement in meters
   - $D$: Down (assumed $0$ for 2D surface navigation)

The 2D rotation matrix from Body frame to Navigation frame parameterized by yaw angle $\psi$ is:

$$R_b^n(\psi) = \begin{bmatrix} \cos\psi & -\sin\psi \\ \sin\psi & \cos\psi \end{bmatrix}$$

---

## 8-State Vector Formulation

The EKF maintains an 8-dimensional state vector $\mathbf{x} \in \mathbb{R}^8$:

$$\mathbf{x} = \begin{bmatrix} N \\ E \\ v_N \\ v_E \\ \psi \\ b_{a_x} \\ b_{a_y} \\ b_{\omega_z} \end{bmatrix} \quad \begin{array}{l} \text{North position (meters)} \\ \text{East position (meters)} \\ \text{North velocity (m/s)} \\ \text{East velocity (m/s)} \\ \text{Vehicle heading / yaw (radians)} \\ \text{Body X-axis accelerometer bias (m/s}^2\text{)} \\ \text{Body Y-axis accelerometer bias (m/s}^2\text{)} \\ \text{Z-axis gyroscope yaw rate bias (rad/s)} \end{array}$$

---

## Process Model & Prediction Step (Time Update)

Given control input $\mathbf{u}_k = \begin{bmatrix} \tilde{a}_{x,k} & \tilde{a}_{y,k} & \tilde{\omega}_{z,k} \end{bmatrix}^T$ sampled from the IMU at time step $\Delta t = 0.1\text{ s}$ ($10\text{ Hz}$ EKF update rate):

### State Transition Equations:

$$\begin{aligned}
\hat{N}_k &= N_{k-1} + v_{N,k-1} \Delta t + \frac{1}{2} a_{N,k-1} \Delta t^2 \\
\hat{E}_k &= E_{k-1} + v_{E,k-1} \Delta t + \frac{1}{2} a_{E,k-1} \Delta t^2 \\
\hat{v}_{N,k} &= v_{N,k-1} + a_{N,k-1} \Delta t \\
\hat{v}_{E,k} &= v_{E,k-1} + a_{E,k-1} \Delta t \\
\hat{\psi}_k &= \psi_{k-1} + (\tilde{\omega}_{z,k} - b_{\omega_z,k-1}) \Delta t \\
\hat{b}_{a_x,k} &= b_{a_x,k-1} \\
\hat{b}_{a_y,k} &= b_{a_y,k-1} \\
\hat{b}_{\omega_z,k} &= b_{\omega_z,k-1}
\end{aligned}$$

where unbiased body accelerations are projected into navigation coordinates:

$$\begin{bmatrix} a_N \\ a_E \end{bmatrix} = R_b^n(\psi) \begin{bmatrix} \tilde{a}_x - b_{a_x} \\ \tilde{a}_y - b_{a_y} \end{bmatrix} = \begin{bmatrix} \cos\psi(\tilde{a}_x - b_{a_x}) - \sin\psi(\tilde{a}_y - b_{a_y}) \\ \sin\psi(\tilde{a}_x - b_{a_x}) + \cos\psi(\tilde{a}_y - b_{a_y}) \end{bmatrix}$$

### State Transition Jacobian ($F = \frac{\partial f}{\partial \mathbf{x}}$):

$$F_k = \begin{bmatrix}
1 & 0 & \Delta t & 0 & f_{15} & -\frac{1}{2}\Delta t^2 \cos\psi & \frac{1}{2}\Delta t^2 \sin\psi & 0 \\
0 & 1 & 0 & \Delta t & f_{25} & -\frac{1}{2}\Delta t^2 \sin\psi & -\frac{1}{2}\Delta t^2 \cos\psi & 0 \\
0 & 0 & 1 & 0 & f_{35} & -\Delta t \cos\psi & \Delta t \sin\psi & 0 \\
0 & 0 & 0 & 1 & f_{45} & -\Delta t \sin\psi & -\Delta t \cos\psi & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & -\Delta t \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

where the heading partial derivatives are:
- $f_{35} = \frac{\partial v_N}{\partial \psi} = \left( -\sin\psi (\tilde{a}_x - b_{a_x}) - \cos\psi (\tilde{a}_y - b_{a_y}) \right) \Delta t$
- $f_{45} = \frac{\partial v_E}{\partial \psi} = \left( \cos\psi (\tilde{a}_x - b_{a_x}) - \sin\psi (\tilde{a}_y - b_{a_y}) \right) \Delta t$
- $f_{15} = \frac{1}{2} \Delta t \cdot f_{35}$
- $f_{25} = \frac{1}{2} \Delta t \cdot f_{45}$

### Covariance Prediction:

$$P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q$$

---

## Measurement Update Models (Correction Step)

### 1. GNSS Position Measurement (Satellite Visible)

When satellite signals are active, GPS provides absolute global positions $\mathbf{z}_{gps} = \begin{bmatrix} N_{gps} & E_{gps} \end{bmatrix}^T$:

$$H_{gps} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}, \quad R_{gps} = \begin{bmatrix} \sigma_{gps}^2 & 0 \\ 0 & \sigma_{gps}^2 \end{bmatrix} \quad (\sigma_{gps} \approx 2.5\text{ m})$$

### 2. AI Velocity Measurement (Active in Blackout & Open Sky)

The BiLSTM model outputs estimated forward speed $\hat{v}_{ai}$. The measurement relationship is:

$$\hat{v}_{ai} = \sqrt{v_N^2 + v_E^2} = v_N \cos\psi + v_E \sin\psi$$

The measurement Jacobian $H_{ai} = \frac{\partial h_{ai}}{\partial \mathbf{x}}$ is:

$$H_{ai} = \begin{bmatrix} 0 & 0 & \cos\psi & \sin\psi & (-v_N \sin\psi + v_E \cos\psi) & 0 & 0 & 0 \end{bmatrix}$$

with measurement noise variance $R_{ai} = \sigma_{v_{ai}}^2 \approx (1.69\text{ m/s})^2 = 2.856\text{ (m/s)}^2$.

### 3. Non-Holonomic Constraint (NHC)

Ground vehicles do not slide laterally or jump vertically under normal driving conditions:

$$v_y^b = -v_N \sin\psi + v_E \cos\psi \approx 0$$

The measurement model $z_{nhc} = 0$ with Jacobian:

$$H_{nhc} = \begin{bmatrix} 0 & 0 & -\sin\psi & \cos\psi & (-v_N \cos\psi - v_E \sin\psi) & 0 & 0 & 0 \end{bmatrix}$$

with tight constraint covariance $R_{nhc} = \sigma_{nhc}^2 \approx (0.1\text{ m/s})^2 = 0.01\text{ (m/s)}^2$.

### 4. Zero-Velocity Update (ZUPT)

When the vehicle comes to a complete halt (detected by low variance in acceleration and gyro signals):

$$\mathbf{z}_{zupt} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies H_{zupt} = \begin{bmatrix} 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix}, \quad R_{zupt} = 10^{-4} \cdot I_{2\times 2}$$

ZUPT directly resets accumulated velocity drift and allows rapid calibration of accelerometer zero-bias.

---

## Outage Management State Machine

```
                      ┌────────────────────────────┐
                      │    STATE 1: GNSS LOCKED    │
                      │  • 1 Hz GNSS Updates       │
                      │  • EKF Calibrates ba, bg   │
                      │  • Covariance P Converges  │
                      └─────────────┬──────────────┘
                                    │
              [ GNSS SNR Drops / Tunnel Entry Detected ]
                                    │
                                    ▼
                      ┌────────────────────────────┐
                      │  STATE 2: TUNNEL BLACKOUT  │
                      │  • Zero-Latency AI Handoff │
                      │  • EKF + v_ai + NHC + ZUPT │
                      │  • Bias-Corrected Gyro Yaw │
                      └─────────────┬──────────────┘
                                    │
              [ GNSS Reacquired / Tunnel Exit Detected ]
                                    │
                                    ▼
                      ┌────────────────────────────┐
                      │  STATE 3: RE-ACQUISITION   │
                      │  • Smooth Trajectory Anchor│
                      │  • Zero Route Freeze / Jump│
                      │  • Recalibrate Covariances │
                      └─────────────┬──────────────┘
                                    │
                                    └────► Returns to STATE 1
```
