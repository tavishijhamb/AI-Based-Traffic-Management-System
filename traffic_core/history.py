import os
import pandas as pd
from datetime import datetime


def append_observations(
    path,
    counts,
    densities,
    observed_at=None
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    now = observed_at or datetime.now()

    rows = []

    for lane in ["North", "South", "East", "West"]:
        rows.append({
            "timestamp": now.isoformat(),
            "lane": lane,
            "vehicle_count": int(counts.get(lane, 0)),
            "minute_of_day": now.hour * 60 + now.minute,
            "day_of_week": now.weekday(),
            "density": densities.get(lane, "LOW")
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False
    )
