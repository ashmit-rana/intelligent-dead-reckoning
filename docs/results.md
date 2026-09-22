# Experimental Results & Benchmark Evaluation

## Benchmark Dataset: Real-World IO-VNBD Drive

All models and estimators were evaluated on the **IO-VNBD (IMU-Odometry Vehicle Navigation Benchmark Dataset)** collected by Coventry University and Warwick University.
- **Route Length:** **$12.63\text{ km}$** ($12,631.01\text{ meters}$)
- **Driving Conditions:** Mixed urban driving, high-speed arterials, intersections, roundabouts, and elevation changes.
- **Sensors:** Uncalibrated smartphone MEMS IMU (100 Hz) mounted in the vehicle cabin.
- **Ground Truth:** Vehicle CAN-Bus wheel speed and high-precision RTK GNSS receiver.

---

## Head-to-Head Performance Summary

The table below presents the quantitative comparison across the three navigation paradigms evaluated over the full $12.63\text{ km}$ trajectory:

| Metric | Raw Dead Reckoning | Classical EKF | Our AI-EKF System | SIH 2026 Target | Pass / Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Speed Estimation RMSE** | Diverges ($> 50\text{ m/s}$) | N/A (accel integration) | **$1.69\text{ m/s}$** | $< 2.0\text{ m/s}$ | **PASSED ✅** |
| **Blackout Outage RMSE** | $12,883.15\text{ m}$ | $113.90\text{ m}$ | **$52.21\text{ m}$** | $< 60.0\text{ m}$ | **PASSED ✅** |
| **Blackout Max Pos Error** | $17,441.21\text{ m}$ | $269.14\text{ m}$ | **$84.74\text{ m}$** | $< 100.0\text{ m}$ | **PASSED ✅ (68.5% drop)** |
| **Blackout Drift Rate** | $3,372.86\%$ | $52.05\%$ | **$16.39\%$** | $< 20.0\%$ | **PASSED ✅** |
| **Overall Route RMSE** | $28,041.79\text{ m}$ | $45.47\text{ m}$ | **$42.31\text{ m}$** | $< 50.0\text{ m}$ | **PASSED ✅** |
| **Overall Route MAE** | $23,765.89\text{ m}$ | $37.43\text{ m}$ | **$36.39\text{ m}$** | $< 40.0\text{ m}$ | **PASSED ✅** |
| **Final Trajectory Error** | $57,850.35\text{ m}$ | $55.21\text{ m}$ | **$55.21\text{ m}$** | $< 60.0\text{ m}$ | **PASSED ✅** |
| **Inference Latency** | N/A | $0.2\text{ ms}$ | **$1.78\text{ ms}$** | $< 10.0\text{ ms}$ | **PASSED ✅ (10 Hz CPU)** |

---

## Detailed Blackout Scenario Analysis

To test robustness under critical tunnel conditions, a synthetic GNSS blackout was injected along the route:
- **Blackout Duration:** $517.10\text{ meters}$ traveled during continuous satellite denial.

```
BLACKOUT DRIFT COMPARISON:
Error (m)
300m │                                          * Classical EKF (269.14 m)
250m │                                    *
200m │                              *
150m │                        *
100m │                  *               * Target Threshold (100 m)
 84m │────────────* (AI-EKF Peak: 84.74 m ✅)
 50m │      *     
  0m └──────────────────────────────────────────────────────────► Distance in Outage (m)
     0m        100m        200m        300m        400m        517m
```

### Key Takeaways from Outage Testing:
1. **Raw Dead Reckoning Disintegration:** Integrating raw uncalibrated smartphone accelerations leads to **$17.44\text{ km}$ of error** over a $517\text{ m}$ outage—proving unguided double integration is completely unusable.
2. **Classical EKF Drift:** While the classical filter tracks bias, slight residual heading and acceleration noise accumulate to **$269.14\text{ meters}$ of maximum error** ($52.05\%$ drift rate).
3. **AI-EKF Bounding:** Direct BiLSTM speed regression + Non-Holonomic Constraints clamp maximum error to **$84.74\text{ meters}$** ($16.39\%$ drift rate)—a **$68.5\%$ reduction in peak drift**.

---

## Metric Formulas & Mathematical Definitions

1. **Root Mean Square Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( (N_i - N_i^{GT})^2 + (E_i - E_i^{GT})^2 \right)}$$

2. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \sqrt{(N_i - N_i^{GT})^2 + (E_i - E_i^{GT})^2}$$

3. **Maximum Position Error:**
   $$\text{Max Error} = \max_{i \in \text{blackout}} \sqrt{(N_i - N_i^{GT})^2 + (E_i - E_i^{GT})^2}$$

4. **Drift Rate Percentage:**
   $$\text{Drift Rate} = \frac{\text{Max Blackout Error}}{\text{Total Distance Traveled during Blackout}} \times 100\% = \frac{84.74\text{ m}}{517.10\text{ m}} \times 100\% = 16.39\%$$

---

## Visual Benchmark Figures

The evaluation pipeline automatically generates high-resolution figures in `results/figures/`:

1. **`trajectory.png`**: Full $12.63\text{ km}$ ground truth route vs. AI-EKF estimated trajectory.
2. **`blackout.png`**: Zoomed-in tunnel blackout segment showing classical EKF divergence vs. AI-EKF drift clamping.
3. **`speed.png`**: BiLSTM predicted speed overlayed against vehicle CAN-bus ground truth velocity.
4. **`training.png`**: Neural network training and validation loss curves showing convergence.
5. **`architecture_diagram_v2.png`**: Full mathematical data flow diagram.
