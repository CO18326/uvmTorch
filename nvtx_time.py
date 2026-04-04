import sqlite3
import sys

def print_nvtx_duration_us(db_path, nvtx_range_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Adjust table name if needed (NVTX_EVENTS / NVTX_PUSHPOP_EVENTS)
    query = """
        SELECT start, end
        FROM NVTX_EVENTS
        WHERE text = ?
        ORDER BY start
        LIMIT 1;
    """

    cursor.execute(query, (nvtx_range_name,))
    row = cursor.fetchone()

    if not row:
        print(f"NVTX range '{nvtx_range_name}' not found!")
        conn.close()
        return

    start_ns, end_ns = row
    duration_us = (end_ns - start_ns) / 1000.0

    print(f"NVTX Range: {nvtx_range_name}")
    print(f"Total Time: {duration_us:.3f} microseconds")

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python nvtx_time.py <sqlite_file> <nvtx_range_name>")
        sys.exit(1)

    db_path = sys.argv[1]
    nvtx_range_name = sys.argv[2]

    print_nvtx_duration_us(db_path, nvtx_range_name)
