import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

today = pd.to_datetime("2025-12-05")  # updated start date

tasks = [
    "Lubricate bearings",
    "Inspect bearings",
    "Replace bearings",
    "Add vibration alerting",
    "Laser alignment",
    "Check coupling",
    "Improve housekeeping",
    "Clean cooling fins",
    "Add airflow monitoring",
    "Inspect transformer taps",
    "Tighten connections",
    "Validate voltage balance",
]

# Offsets to stagger tasks realistically
offset_days = [0, 0, 3, 2, 8, 12, 1, 4, 5, 8, 10, 12]
durations = [2,3,5,4,4,2,3,1,3,2,2,3]

start_dates = [today + pd.Timedelta(days=d) for d in offset_days]

fig, ax = plt.subplots(figsize=(14, 7))

for i, task in enumerate(tasks):
    ax.barh(task, durations[i], left=start_dates[i], height=0.5)

ax.set_title("FMEA Action Plan – Gantt Chart (Starting December 5, 2025)", fontsize=16)
ax.set_xlabel("Date")

ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

today = pd.to_datetime("2025-12-05")

tasks = [
    "Weekly anomaly review",
    "Monthly vibration analysis",
    "Monthly current draw review",
    "30-day sensor audit",
    "Quarterly alignment check",
    "90-day threshold recalibration",
    "Annual rebuild assessment",
]

# Compressed offsets so the chart resembles the FMEA timeline
# (first occurrence only, visually aligned)
start_offsets = [0, 3, 3, 3, 7, 7, 12]
durations = [1, 1, 1, 1, 1, 1, 2]

start_dates = [today + pd.Timedelta(days=d) for d in start_offsets]

fig, ax = plt.subplots(figsize=(14,7))

for i, task in enumerate(tasks):
    ax.barh(task, durations[i], left=start_dates[i], height=0.5)

ax.set_title("Predictive Maintenance Routine – Gantt Chart (Starting December 5, 2025)", fontsize=16)
ax.set_xlabel("Date")

# Match FMEA chart formatting
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()