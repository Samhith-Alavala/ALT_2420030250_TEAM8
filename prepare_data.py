
import pandas as pd
import numpy as np
from pathlib import Path

# FedFit - Prepare WISDM time-series windows

DATA_PATH = Path("data/wisdm_clean.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 100       # 5 seconds at 20 Hz
STEP_SIZE = 50          # 50% overlap

FEATURES = ["x_acc", "y_acc", "z_acc"]
LABELS = {
    "Downstairs": 0,
    "Jogging": 1,
    "Sitting": 2,
    "Standing": 3,
    "Upstairs": 4,
    "Walking": 5
}

print("Loading cleaned WISDM dataset...")
df = pd.read_csv(DATA_PATH)
df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)

# Store windows, labels and user IDs separately
X_windows = []
y_labels = []
user_ids = []

print("Creating time-series windows...")

for user_id, user_data in df.groupby("user", sort=True):
    user_data = user_data.reset_index(drop=True)

    # Do not create windows across activity boundaries
    activities = user_data["activity"].to_numpy()
    sensor_values = user_data[FEATURES].to_numpy(dtype=np.float32)

    start = 0

    while start + WINDOW_SIZE <= len(user_data):
        end = start + WINDOW_SIZE
        window_activities = activities[start:end]

        # Keep only windows where all 100 readings have one label
        if np.all(window_activities == window_activities[0]):
            activity = window_activities[0]

            X_windows.append(sensor_values[start:end])
            y_labels.append(LABELS[activity])
            user_ids.append(int(user_id))

        start += STEP_SIZE

print("Converting windows to arrays...")

X = np.asarray(X_windows, dtype=np.float32)
y = np.asarray(y_labels, dtype=np.int64)
users = np.asarray(user_ids, dtype=np.int64)

np.save(OUTPUT_DIR / "X.npy", X)
np.save(OUTPUT_DIR / "y.npy", y)
np.save(OUTPUT_DIR / "users.npy", users)

# Save a readable mapping for reference
mapping = pd.DataFrame([
    {"label": label, "activity": activity}
    for activity, label in LABELS.items()
])
mapping.to_csv(OUTPUT_DIR / "label_mapping.csv", index=False)

print("\nWindow preparation completed!")
print("Window data shape:", X.shape)
print("Label shape:", y.shape)
print("User ID shape:", users.shape)
print("Number of windows:", len(X))
print("Number of users:", len(np.unique(users)))

print("\nWindows per activity:")
for label, count in zip(*np.unique(y, return_counts=True)):
    print(f"{label} - {mapping.loc[mapping['label'] == label, 'activity'].iloc[0]}: {count}")

print("\nSaved files in:", OUTPUT_DIR)
print("- X.npy: sensor windows")
print("- y.npy: activity labels")
print("- users.npy: user/client IDs")
print("- label_mapping.csv: activity label reference")
