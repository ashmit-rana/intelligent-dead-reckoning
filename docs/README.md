# Intelligent Dead Reckoning System — Documentation Hub

Welcome to the technical documentation for the **AI-ML Based Intelligent Dead Reckoning System** (Smart India Hackathon 2026, Problem Statement: `SIH26168`, Theme: *Smart Vehicles / Space Tech*).

This project provides an infrastructure-free, software-only dead reckoning navigation engine that guarantees continuous, high-accuracy vehicle positioning during prolonged GNSS/NavIC satellite blackouts (e.g., tunnels, urban canyons, underground facilities) using standard smartphone IMU sensors fused with deep neural networks and an 8-State Extended Kalman Filter (EKF).

---

## Documentation Index

| Document | Description | Key Topics |
| :--- | :--- | :--- |
| **[Overview](overview.md)** | Executive summary, problem definition, and solution rationale. | GNSS vulnerability, quadratic error explosion, core innovations, value proposition. |
| **[Architecture](architecture.md)** | End-to-end system design and mathematical foundations. | Sensor ingestion, 8-State EKF formulation, Jacobians, non-holonomic constraints, ZUPT. |
| **[Models](models.md)** | Deep learning velocity regression model. | BiLSTM architecture, temporal attention, windowing, feature normalization, training loss. |
| **[Pipeline](pipeline.md)** | Data flow, signal processing, and coordinate transformations. | Geodetic to NED conversion, DCM gravity stripping, Butterworth filtering, rolling buffers. |
| **[Results & Benchmarks](results.md)** | Empirical verification on real-world vehicle telemetry. | IO-VNBD dataset (12.63 km drive), Raw DR vs. EKF vs. AI-EKF, blackout metrics, drift reduction. |
| **[Development Guide](development.md)** | Setup, project structure, training, and execution workflow. | Virtual environments, dependencies, running pipelines, evaluation scripts, code quality. |
| **[Deployment & Integration](deployment.md)** | Edge optimization, SDK design, and ISRO NavIC integration. | Quantization (< 2.5 MB), sub-2 ms CPU latency, Android NDK / iOS SDK, NMEA 0183 protocol. |

---

## Quick Overview of the System

```
[ Smartphone MEMS IMU ] (100 Hz Accel & Gyro)
           │
           ▼
[ Signal Preprocessing ] (Gravity Stripping via DCM + Butterworth Filter + 1s Window)
           │
           ├───► [ BiLSTM Neural Speed Regressor ] ───► Predicted Forward Velocity (m/s)
           │                                                    │
           ▼                                                    ▼
[ 8-State Extended Kalman Filter ] ◄────────────────────────────┘
     ▲  State: [N, E, vN, vE, ψ, ba_x, ba_y, bg_z]
     │  Constraints: Non-Holonomic (vy ≈ 0) + ZUPT (zero speed at stops)
     │
[ GNSS / NavIC Receiver ] (1 Hz satellite anchor when visible; auto-calibrates bias)
           │
           ▼
[ Real-Time 10 Hz Continuous Trajectory HUD ] (< 1.8 ms latency per frame)
```

---

## Key Performance Highlights

- **BiLSTM Velocity RMSE:** **1.69 m/s** *(SIH Target: < 2.0 m/s ✅)*
- **Max Blackout Error:** **84.74 m** *(68.5% reduction vs. 269.14 m classical EKF ✅)*
- **Blackout Drift Rate:** **16.39%** *(vs. 52.05% classical EKF & 3,372% raw DR ✅)*
- **Overall Route RMSE:** **42.31 m** over full 12.63 km trajectory *(SIH Target: < 50.0 m ✅)*
- **On-Device Edge Latency:** **< 1.8 ms** per inference frame on smartphone CPU *(Zero GPU dependency)*
- **Infrastructure Cost:** **₹0** *(100% software; zero roadside RF beacons or subterranean repeaters)*

---

## Core Technologies

- **Core Languages:** Python 3.9+, C++17, Kotlin, Swift
- **Deep Learning:** PyTorch 2.0, ONNX Runtime (INT8 Quantization)
- **State Estimation:** 8-State Extended Kalman Filter, NumPy, SciPy, FilterPy
- **Protocols & Standards:** ISRO NavIC, NMEA 0183, WGS-84 / Local NED
- **Dataset Benchmarks:** IO-VNBD (Coventry/Warwick 12.63 km drive with CAN-bus ground truth)
