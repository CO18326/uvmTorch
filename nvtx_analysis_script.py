import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect("deepspeed_satge2_670_1")

# Load NVTX ranges
nvtx = pd.read_sql_query(
    "SELECT text, start, end FROM NVTX_EVENTS;",
    conn
)

# Filter NVTX ranges (Step / Forward / Backward / Optimizer)
nvtx = nvtx[nvtx["text"].str.contains("Step", na=False)]
# nvtx = nvtx[nvtx["text"].str.contains("Forward", na=False)]
# nvtx = nvtx[nvtx["text"].str.contains("Backward", na=False)]
# nvtx = nvtx[nvtx["text"].str.contains("Optimizer", na=False)]

# Load memcpy activities
memcpy = pd.read_sql_query("""
    SELECT start, end, bytes, copyKind
    FROM CUPTI_ACTIVITY_KIND_MEMCPY;
""", conn)

results = []

for _, row in nvtx.iterrows():
    step = row["text"]
    start, end = row["start"], row["end"]

    range_time_ns = end - start

    # Memcpy ops fully inside NVTX range
    in_range = memcpy[
        (memcpy["start"] >= start) &
        (memcpy["end"] <= end)
    ].copy()

    # Duration per memcpy
    in_range["duration_ns"] = in_range["end"] - in_range["start"]

    # Bytes
    total_bytes = in_range["bytes"].sum()
    h2d_bytes = in_range[in_range["copyKind"] == 1]["bytes"].sum()
    d2h_bytes = in_range[in_range["copyKind"] == 2]["bytes"].sum()
    d2d_bytes = in_range[in_range["copyKind"] == 8]["bytes"].sum()

    # Time
    total_memcpy_time_ns = in_range["duration_ns"].sum()
    h2d_time_ns = in_range[in_range["copyKind"] == 1]["duration_ns"].sum()
    d2h_time_ns = in_range[in_range["copyKind"] == 2]["duration_ns"].sum()
    d2d_time_ns = in_range[in_range["copyKind"] == 8]["duration_ns"].sum()

    results.append({
        "step": step,

        # NVTX range timing
        "range_time_ms": range_time_ns / 1e6,

        # Memcpy timing
        "total_memcpy_time_ms": total_memcpy_time_ns / 1e6,
        "h2d_time_ms": h2d_time_ns / 1e6,
        "d2h_time_ms": d2h_time_ns / 1e6,
        "d2d_time_ms": d2d_time_ns / 1e6,

        # Memcpy bytes
        "total_bytes": total_bytes,
        "h2d_bytes": h2d_bytes,
        "d2h_bytes": d2h_bytes,
        "d2d_bytes": d2d_bytes,

        # Counts
        "num_transfers": len(in_range),

        # Useful ratio
        "memcpy_pct_of_range":
            (total_memcpy_time_ns / range_time_ns * 100)
            if range_time_ns > 0 else 0.0
    })

df = pd.DataFrame(results)
df.to_csv("deepspeed_satge2_670_1.csv", index=False)

#print("Saved: deepspeed_910_stage_2.csv")
