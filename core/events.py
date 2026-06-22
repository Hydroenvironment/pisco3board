from __future__ import annotations

import pandas as pd


def detect_precipitation_events(df: pd.DataFrame, value_col: str = "precipitation", threshold: float = 10.0, min_duration_days: int = 1, max_gap_days: int = 0) -> pd.DataFrame:
    out = df[["time", value_col]].copy()
    out["time"] = pd.to_datetime(out["time"])
    out = out.sort_values("time").reset_index(drop=True)
    out["is_event_day"] = out[value_col] >= threshold
    if max_gap_days > 0:
        flags = out["is_event_day"].to_numpy().copy()
        n = len(flags)
        i = 0
        while i < n:
            if flags[i]:
                j = i + 1
                while j < n and not flags[j]:
                    j += 1
                gap = j - i - 1
                if j < n and gap <= max_gap_days:
                    flags[i + 1:j] = True
                i = j
            else:
                i += 1
        out["is_event_day"] = flags
    events = []
    in_event = False
    start_idx = None
    flags = out["is_event_day"].to_numpy()
    for i, flag in enumerate(flags):
        if flag and not in_event:
            in_event = True
            start_idx = i
        elif not flag and in_event:
            events.append((start_idx, i - 1))
            in_event = False
    if in_event:
        events.append((start_idx, len(out) - 1))
    rows = []
    for k, (start, end) in enumerate(events, start=1):
        segment = out.iloc[start:end + 1]
        exceedance_segment = segment[segment[value_col] >= threshold]
        duration = len(segment)
        if duration < min_duration_days:
            continue
        rows.append({
            "event_id": k,
            "start": segment["time"].iloc[0],
            "end": segment["time"].iloc[-1],
            "duration_days": duration,
            "event_total_mm": float(segment[value_col].sum()),
            "event_max_1day_mm": float(segment[value_col].max()),
            "event_mean_mm_day": float(segment[value_col].mean()),
            "exceedance_days": int(len(exceedance_segment)),
            "threshold_mm_day": float(threshold),
        })
    return pd.DataFrame(rows)
