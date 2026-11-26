import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

CSV_PATH = "data/traffic_history.csv"
MODEL_PATH = "models/congestion_model.joblib"

os.makedirs("models", exist_ok=True)

LANES = ["North", "South", "East", "West"]

# 15-second readings:
# 4 readings = 1 minute
# 20 readings = 5 minutes

FUTURE_STEPS = 20

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"CSV file not found: {CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

required = [
    "timestamp",
    "lane",
    "vehicle_count"
]

for column in required:
    if column not in df.columns:
        raise ValueError(
            f"Missing required CSV column: {column}"
        )

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df["vehicle_count"] = pd.to_numeric(
    df["vehicle_count"],
    errors="coerce"
)

df["lane"] = df["lane"].astype(str).str.strip()

df = df.dropna(
    subset=["timestamp", "vehicle_count"]
)

df = df[df["lane"].isin(LANES)]

df = df[df["vehicle_count"] >= 0]

df = df.sort_values("timestamp")

print("Total valid observations:", len(df))
print("\nObservations by lane:")
print(df["lane"].value_counts())

# --------------------------------------------------
# 3. PREPARE TIME FEATURES
# --------------------------------------------------

df["minute_of_day"] = (
    df["timestamp"].dt.hour * 60
    + df["timestamp"].dt.minute
)

df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)

# --------------------------------------------------
# 4. CREATE HISTORICAL FEATURES
# --------------------------------------------------

all_rows = []

for lane in LANES:

    lane_df = df[
        df["lane"] == lane
    ].copy()

    lane_df = lane_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if len(lane_df) <= FUTURE_STEPS + 20:
        print(
            f"Not enough data for {lane}. Skipping."
        )
        continue

    lane_df["count_now"] = (
        lane_df["vehicle_count"]
    )

    # Previous readings:
    lane_df["count_1min_ago"] = (
        lane_df["vehicle_count"].shift(4)
    )

    lane_df["count_2min_ago"] = (
        lane_df["vehicle_count"].shift(8)
    )

    lane_df["count_3min_ago"] = (
        lane_df["vehicle_count"].shift(12)
    )

    lane_df["count_4min_ago"] = (
        lane_df["vehicle_count"].shift(16)
    )

    # Average over the last 5 minutes:
    lane_df["average_5min"] = (
        lane_df["vehicle_count"]
        .rolling(window=20)
        .mean()
    )

    # Target: density 5 minutes into the future.
    future_count = (
        lane_df["vehicle_count"]
        .shift(-FUTURE_STEPS)
    )

    lane_df["target"] = np.select(
        [
            future_count < 5,
            future_count < 12
        ],
        [
            "LOW",
            "MEDIUM"
        ],
        default="HIGH"
    )

    lane_df["lane"] = lane

    all_rows.append(lane_df)

# --------------------------------------------------
# 5. COMBINE ALL LANES
# --------------------------------------------------

if not all_rows:
    raise ValueError(
        "No lane has enough observations. "
        "Collect more traffic data."
    )

data = pd.concat(
    all_rows,
    ignore_index=True
)

features = [
    "lane",
    "count_now",
    "count_1min_ago",
    "count_2min_ago",
    "count_3min_ago",
    "count_4min_ago",
    "average_5min",
    "minute_of_day",
    "day_of_week"
]

data = data.dropna(
    subset=features + ["target"]
)

# One-hot encode the lane.
X = pd.get_dummies(
    data[features],
    columns=["lane"]
)

# Ensure all four lane columns exist.
for lane in LANES:
    column = f"lane_{lane}"

    if column not in X.columns:
        X[column] = 0

# Keep a consistent feature order.
feature_columns = sorted(X.columns)

X = X[feature_columns]

y = data["target"]

# --------------------------------------------------
# 6. CHRONOLOGICAL TRAIN/TEST SPLIT
# --------------------------------------------------

data = data.sort_values(
    "timestamp"
).reset_index(drop=True)

# Re-create X and y in chronological order.
X = pd.get_dummies(
    data[features],
    columns=["lane"]
)

for lane in LANES:
    column = f"lane_{lane}"

    if column not in X.columns:
        X[column] = 0

feature_columns = sorted(X.columns)
X = X[feature_columns]
y = data["target"]

split = int(len(data) * 0.8)

if split == 0 or split == len(data):
    raise ValueError("Not enough data for train/test split.")

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# --------------------------------------------------
# 7. TRAIN RANDOM FOREST
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

# --------------------------------------------------
# 8. EVALUATE MODEL
# --------------------------------------------------

predictions = model.predict(X_test)

print("\nMODEL EVALUATION")
print("-------------------------")

print(
    "Accuracy:",
    round(
        accuracy_score(y_test, predictions) * 100,
        2
    ),
    "%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["LOW", "MEDIUM", "HIGH"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["LOW", "MEDIUM", "HIGH"]
    )
)

# --------------------------------------------------
# 9. SAVE MODEL
# --------------------------------------------------

joblib.dump(
    {
        "model": model,
        "features": feature_columns,
        "lanes": LANES,
        "future_steps": FUTURE_STEPS
    },
    MODEL_PATH
)

print("\nTraining completed successfully!")
print("Model saved at:", MODEL_PATH)
