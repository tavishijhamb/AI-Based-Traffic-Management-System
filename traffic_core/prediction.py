import os
import joblib
import pandas as pd

from collections import defaultdict, deque

MODEL_PATH = "models/congestion_model.joblib"

LANES = ["North", "South", "East", "West"]

def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    bundle = joblib.load(MODEL_PATH)

    return bundle

class CongestionForecaster:

    def __init__(self, bundle):

        self.model = bundle["model"]
        self.features = bundle["features"]

        self.history = defaultdict(
            lambda: deque(maxlen=20)
        )

    def predict(self, counts, observation_time):

        results = {}

        for lane in LANES:

            current_count = max(
                0,
                int(counts.get(lane, 0))
            )

            self.history[lane].append(
                current_count
            )

            readings = list(
                self.history[lane]
            )

            if len(readings) < 20:

                results[lane] = (
                    f"Collecting history "
                    f"({len(readings)}/20)"
                )

                continue

            features = {

                "count_now": readings[-1],

                "count_1min_ago": readings[-5],

                "count_2min_ago": readings[-9],

                "count_3min_ago": readings[-13],

                "count_4min_ago": readings[-17],

                "average_5min": (
                    sum(readings) / len(readings)
                ),

                "minute_of_day": (
                    observation_time.hour * 60
                    + observation_time.minute
                ),

                "day_of_week": (
                    observation_time.weekday()
                )
            }

            for direction in LANES:

                features[
                    f"lane_{direction}"
                ] = int(direction == lane)

            input_df = pd.DataFrame(
                [features]
            )

            input_df = input_df.reindex(
                columns=self.features,
                fill_value=0
            )

            prediction = self.model.predict(
                input_df
            )[0]

            results[lane] = prediction

        return results