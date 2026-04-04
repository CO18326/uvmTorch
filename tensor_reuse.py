import pandas as pd
import numpy as np

def analyze_tensor_usage(csv_file):
    df = pd.read_csv(csv_file)

    # Sort by time
    df = df.sort_values("timestamp_ns")

    results = []

    grouped = df.groupby("tensor_id")

    for tensor_id, group in grouped:
        timestamps = group["timestamp_ns"].values
        count = len(timestamps)

        avg_time = np.mean(timestamps)

        if count > 1:
            gaps = np.diff(timestamps)
            avg_gap = np.mean(gaps)
        else:
            avg_gap = np.nan

        results.append({
            "tensor_id": tensor_id,
            "count": count,
            "avg_timestamp_ns": avg_time,
            "avg_gap_ns": avg_gap
        })

    result_df = pd.DataFrame(results)

    # ✅ Global metrics
    global_avg_gap = result_df["avg_gap_ns"].mean(skipna=True)
    global_avg_count = result_df["count"].mean()   # 🔥 THIS is new

    return result_df, global_avg_gap, global_avg_count


if __name__ == "__main__":
    result_df, global_avg_gap, global_avg_count = analyze_tensor_usage("tensor_log_pytorch_change_2.csv")

    result_df = result_df.sort_values("count", ascending=False)

    print("\nPer Tensor Stats:\n")
    print(result_df.to_string(index=False))

    print("\nGlobal Average Gap (ns):", global_avg_gap)
    print("Global Average Count:", global_avg_count)
