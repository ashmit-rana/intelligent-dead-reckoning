# Development & Engineering Guide

## System Prerequisites

- **Operating System:** Linux (Ubuntu 20.04+), macOS (Apple Silicon / Intel), or Windows 10/11 with WSL2
- **Python Version:** Python 3.9, 3.10, or 3.11
- **Hardware Requirements:**
  - **Training:** 8 GB RAM minimum (GPU optional; training takes < 3 minutes on CPU)
  - **Inference:** Any standard CPU (x86_64 or ARM64 / Apple Silicon)

---

## Quickstart Setup

### 1. Clone Repository & Setup Virtual Environment

```bash
# Clone repository
git clone https://github.com/ashmit-rana/intelligent-dead-reckoning.git
cd "Dead Reckoning System"

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows (cmd/powershell):
# .\venv\Scripts\activate

# Upgrade pip & install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dependencies (`requirements.txt`)

```text
torch>=2.0.0
numpy>=1.23.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
filterpy>=1.4.5
tqdm>=4.65.0
requests>=2.28.0
```

---

## Running the Complete Pipeline

To run the end-to-end telemetry pipeline (dataset verification, feature extraction, BiLSTM velocity training, 8-State EKF estimation, and figure generation):

```bash
# Execute the master pipeline
python3 run_pipeline.py
```

### Expected Output Summary:

```text
======================================================================
     AI-ML BASED INTELLIGENT DEAD RECKONING SYSTEM PIPELINE
======================================================================
[1/5] Checking and loading dataset...
      Loaded 12.63 km IO-VNBD telemetry drive.
[2/5] Running signal preprocessing & gravity stripping...
      Applied 4th-order Butterworth low-pass filter (fc=15Hz).
[3/5] Training BiLSTM Speed Estimator (50 Epochs)...
      Training completed. Best Val Loss: 0.8421. Speed RMSE: 1.69 m/s.
[4/5] Executing 8-State Extended Kalman Filter...
      Simulating 517 m tunnel blackout...
      - Raw DR Blackout Max Error: 17,441.21 m
      - Classical EKF Blackout Max Error: 269.14 m
      - Our AI-EKF Blackout Max Error: 84.74 m (68.5% reduction!)
[5/5] Generating evaluation metrics & high-res figures...
      Saved results/metrics.json
      Saved results/figures/trajectory.png
      Saved results/figures/blackout.png
      Saved results/figures/speed.png
      Saved results/figures/training.png
======================================================================
 PIPELINE COMPLETE: ALL SIH 2026 BENCHMARK TARGETS PASSED ✅
======================================================================
```

---

## Modular Component Execution

Each component in `src/` can be independently tested and inspected:

### 1. Data Loader & Preprocessing
```bash
python3 src/data_loader.py
```
*Parses raw CSV telemetry files, aligns sample rates, and extracts synchronized GPS + IMU timestamps.*

### 2. Signal Processing & Frame Alignment
```bash
python3 src/signal_processing.py
```
*Executes Direction Cosine Matrix (DCM) rotation, strips gravity, and applies 4th-order Butterworth filtering.*

### 3. BiLSTM Model Training & Inference
```bash
python3 src/model.py
```
*Constructs rolling windows ($W=100$), fits standard scalers, trains the PyTorch model, and saves weights to `models/speed_estimator_real.pt`.*

### 4. 8-State EKF State Estimator
```bash
python3 src/ekf.py
```
*Runs time updates, GNSS corrections, non-holonomic constraints, and ZUPT state resets.*

### 5. Benchmark Evaluation & Plotting
```bash
python3 src/evaluation.py
```
*Computes RMSE, MAE, Max Outage Error, drift percentages, and plots comparison charts.*

---

## Directory Structure

```
Dead Reckoning System/
├── README.md                      # Main project overview
├── requirements.txt               # Python package dependencies
├── run_pipeline.py                # Master execution pipeline
├── download_real_dataset.py       # IO-VNBD dataset fetching script
├── data/
│   ├── raw/                       # Raw sensor CSV files (S-S1, S-S2, V-S1, V-S2)
│   └── processed/                 # Synchronized and normalized feature arrays
├── docs/                          # Technical documentation hub
│   ├── README.md                  # Docs index
│   ├── overview.md                # Motivation & problem statement
│   ├── architecture.md            # System architecture & EKF math
│   ├── models.md                  # BiLSTM neural model specs
│   ├── results.md                 # Empirical benchmarks & metrics
│   ├── development.md             # Developer setup & testing guide
│   ├── pipeline.md                # Data flow & signal conditioning
│   └── deployment.md              # Edge SDK & NavIC integration
├── models/
│   ├── speed_estimator.pt         # Synthetic baseline model weights
│   └── speed_estimator_real.pt    # Real IO-VNBD trained weights
├── results/
│   ├── metrics.json               # Quantitative benchmark metrics
│   └── figures/                   # High-res performance graphs
│       ├── trajectory.png         # Full route ground truth vs AI-EKF
│       ├── blackout.png           # Tunnel outage drift comparison
│       ├── speed.png              # Speed prediction vs CAN-bus ground truth
│       ├── training.png           # Neural training loss curves
│       └── architecture_diagram.png # Modular system dataflow
└── src/
    ├── __init__.py                # Package initialization
    ├── data_loader.py             # CSV parsing and synchronization
    ├── signal_processing.py       # DCM rotation and Butterworth filtering
    ├── model.py                   # BiLSTM PyTorch architecture and training
    ├── ekf.py                     # 8-State Extended Kalman Filter
    └── evaluation.py              # Metric calculation and figure plotting
```
