import pandas as pd
import numpy as np

def compute_usage_rate(csv_file):
    df = pd.read_csv(csv_file)

    # Sort by time
    df = df.sort_values("timestamp_ns")

    start_time = df["timestamp_ns"].min()
    end_time = df["timestamp_ns"].max()

    total_time_ns = end_time - start_time
    total_time_sec = total_time_ns / 1e9

    results = []

    grouped = df.groupby("tensor_id")

    for tensor_id, group in grouped:
        count = len(group)

        # usage rate (per second)
        usage_rate = count / total_time_sec if total_time_sec > 0 else 0

        # avg gap (reuse interval)
        timestamps = group["timestamp_ns"].values
        if len(timestamps) > 1:
            avg_gap = np.mean(np.diff(timestamps))
        else:
            avg_gap = np.nan

        results.append({
            "tensor_id": tensor_id,
            "count": count,
            "usage_rate_per_sec": usage_rate,
            "avg_gap_ns": avg_gap
        })

    result_df = pd.DataFrame(results)

    # Global usage rate
    total_events = len(df)
    global_usage_rate = total_events / total_time_sec

    return result_df, global_usage_rate


if __name__ == "__main__":
    result_df, global_rate = compute_usage_rate("tensor_log_pytorch_change_3.csv")

    # sort by usage rate
    result_df = result_df.sort_values("usage_rate_per_sec", ascending=False)

    print("\nPer Tensor Usage Rate:\n")
    print(result_df.to_string(index=False))

    print("\nGlobal Usage Rate (events/sec):", global_rate)
