# ===============================================================
# IMPORTS
# ===============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# LOAD DATA
# ===============================================================
df = pd.read_csv("data/pdm_sensor_dataset_multiactive.csv")

# Split by motor
motor_a = df[df["asset"] == "Motor_A"].reset_index(drop=True)
motor_b = df[df["asset"] == "Motor_B"].reset_index(drop=True)

# ===============================================================
# 2×2 PREDICTIVE MAINTENANCE DASHBOARD
# ===============================================================
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Predictive Maintenance Dashboard", fontsize=22, fontweight="bold")

# ==============================
# 1. VIBRATION OVER TIME
# ==============================
axs[0, 0].plot(motor_a["day"], motor_a["motor_vibration_mm_s"], color="#d98600", label="Motor A")
axs[0, 0].plot(motor_b["day"], motor_b["motor_vibration_mm_s"], color="#4aa8ff", label="Motor B")
axs[0, 0].set_title("Vibration Over Time – Motor A vs Motor B", fontsize=14)
axs[0, 0].set_xlabel("Time (days)")
axs[0, 0].set_ylabel("Vibration (mm/s)")
axs[0, 0].grid(True, linestyle="--", alpha=0.4)
axs[0, 0].legend()

# ==============================
# 2. ANOMALY SCORE OVER TIME
# ==============================
axs[0, 1].plot(motor_a["day"], motor_a["anomaly_score"], color="#d98600")
axs[0, 1].set_title("Predicted Anomaly Score Over Time", fontsize=14)
axs[0, 1].set_xlabel("Time (days)")
axs[0, 1].set_ylabel("Anomaly Score")
axs[0, 1].grid(True, linestyle="--", alpha=0.4)

# ==============================
# 3. CURRENT DRAW
# ==============================
axs[1, 0].plot(motor_a["day"], motor_a["motor_current_A"], color="#d98600", label="Motor A")
axs[1, 0].plot(motor_b["day"], motor_b["motor_current_A"], color="#4aa8ff", label="Motor B")
axs[1, 0].set_title("Current Draw Over Time – Motor A vs Motor B", fontsize=14)
axs[1, 0].set_xlabel("Time (days)")
axs[1, 0].set_ylabel("Current Draw (A)")
axs[1, 0].grid(True, linestyle="--", alpha=0.4)
axs[1, 0].legend()

# ==============================
# 4. TEMPERATURE
# ==============================
axs[1, 1].plot(motor_a["day"], motor_a["motor_temperature_C"], color="#d98600", label="Motor A")
axs[1, 1].plot(motor_b["day"], motor_b["motor_temperature_C"], color="#4aa8ff", label="Motor B")
axs[1, 1].set_title("Temperature Over Time – Motor A vs Motor B", fontsize=14)
axs[1, 1].set_xlabel("Time (days)")
axs[1, 1].set_ylabel("Temperature (°C)")
axs[1, 1].grid(True, linestyle="--", alpha=0.4)
axs[1, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ===============================================================
# ANOMALY SCORE CURVE + 95% CI + FAILURE PREDICTION (NUMPY ONLY)
# ===============================================================

motor = motor_a.copy()
days = motor["day"].values
anom = motor["anomaly_score"].values

# ---- NumPy Linear Regression (y = m*x + b) ----
m, b = np.polyfit(days, anom, 1)
pred = m * days + b

# Predict failure when anomaly reaches 1.0
failure_threshold = 1.0
predicted_failure_day = (failure_threshold - b) / m

# ---- 95% Confidence Interval ----
rolling_mean = motor["anomaly_score"].rolling(5, min_periods=1).mean()
rolling_std = motor["anomaly_score"].rolling(5, min_periods=1).std().fillna(0)
upper_ci = rolling_mean + 1.96 * rolling_std
lower_ci = rolling_mean - 1.96 * rolling_std

# ---- Plot: Anomaly Score Curve ----
plt.figure(figsize=(14, 8))

plt.plot(days, anom, color="#d98600", label="Anomaly Score")
plt.plot(days, rolling_mean, color="#4aa8ff", linestyle="--", label="Rolling Mean")
plt.fill_between(days, lower_ci, upper_ci, color="#4aa8ff", alpha=0.2, label="95% CI")
plt.plot(days, pred, color="green", linestyle="--", label="Trend (Prediction)")

# Failure annotation
plt.axvline(predicted_failure_day, color="red", linestyle="--", linewidth=2)
plt.text(predicted_failure_day + 1, failure_threshold,
         f"Predicted Failure ≈ Day {predicted_failure_day:.1f}",
         fontsize=12, color="red")

plt.title("Anomaly Score Curve with 95% Confidence Interval + Predicted Failure Day (NumPy Regression)", fontsize=18)
plt.xlabel("Day")
plt.ylabel("Anomaly Score")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# ===============================================================
# PREDICTED vs ACTUAL FAILURE COMPARISON (NUMPY REGRESSION)
# ===============================================================

plt.figure(figsize=(14, 8))

plt.plot(days, anom, color="#d98600", label="Actual Anomaly Score")
plt.plot(days, pred, color="#4aa8ff", linestyle="--", label="Predicted Trend")

plt.axhline(failure_threshold, color="red", linestyle="--", label="Failure Threshold")
plt.axvline(predicted_failure_day, color="red", linestyle="--")

plt.title("Predicted vs Actual Failure Comparison (NumPy Regression)", fontsize=18)
plt.xlabel("Day")
plt.ylabel("Anomaly Score")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()