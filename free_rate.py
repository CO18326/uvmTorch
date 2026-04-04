import pandas as pd
import numpy as np

def analyze_address_usage(csv_file):
    df = pd.read_csv(csv_file)

    # sort by time
    df = df.sort_values("time_stamp")

    start_time = df["time_stamp"].min()
    end_time = df["time_stamp"].max()

    total_time_ns = end_time - start_time
    total_time_sec = total_time_ns / 1e9

    results = []

    grouped = df.groupby("address")

    for addr, group in grouped:
        timestamps = group["time_stamp"].values
        count = len(timestamps)

        # usage rate
        usage_rate = count / total_time_sec if total_time_sec > 0 else 0

        # avg gap
        if count > 1:
            gaps = np.diff(timestamps)
            avg_gap = np.mean(gaps)
        else:
            avg_gap = np.nan

        # total bytes accessed (optional but useful)
        total_bytes = group["size"].sum()

        results.append({
            "address": addr,
            "count": count,
            "usage_rate_per_sec": usage_rate,
            "avg_gap_ns": avg_gap,
            "total_bytes": total_bytes
        })

    result_df = pd.DataFrame(results)

    # global stats
    total_events = len(df)
    global_usage_rate = total_events / total_time_sec

    return result_df, global_usage_rate


if __name__ == "__main__":
    result_df, global_rate = analyze_address_usage("free_2b_3600.csv")

    # sort by usage rate
    result_df = result_df.sort_values("usage_rate_per_sec", ascending=False)

    print("\nPer Address Stats:\n")
    print(result_df.to_string(index=False))

    print("\nGlobal Usage Rate (events/sec):", global_rate)
