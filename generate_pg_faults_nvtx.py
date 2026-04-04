import sqlite3
import csv
import sys

def export_faults_for_nvtx_range(db_path, output_csv, nvtx_range_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Step 1: Get NVTX start & end timestamp
    nvtx_query = f"""
        SELECT start, end
        FROM NVTX_EVENTS
        WHERE text = '{nvtx_range_name}'
        ORDER BY start
        LIMIT 1;
    """

    cursor.execute(nvtx_query)
    nvtx_row = cursor.fetchone()

    if not nvtx_row:
        print(f"NVTX range '{nvtx_range_name}' not found!")
        conn.close()
        return

    nvtx_start, nvtx_end = nvtx_row

    print(f"NVTX Range Found:")
    print(f"Start: {nvtx_start}")
    print(f"End  : {nvtx_end}")

    # Step 2: Extract page faults within range
    fault_query = f"""
        SELECT start, address, faultAccessType, numberOfPageFaults
        FROM CUDA_UM_GPU_PAGE_FAULT_EVENTS
        WHERE start >= {nvtx_start}
        AND start <= {nvtx_end}
        ORDER BY start;
    """

    cursor.execute(fault_query)

    total_faults = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "fault_address_hex", "access_type", "number of page faults"])

        for row in cursor:
            timestamp_ns = row[0]
            address_hex = hex(row[1])
            access_type = row[2]
            num_pg_fault=row[3]
            writer.writerow([timestamp_ns, address_hex,access_type,num_pg_fault])
            total_faults += 1

    conn.close()

    print(f"\nCSV written to {output_csv}")
    print(f"Total Page Faults inside '{nvtx_range_name}': {total_faults}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python export_faults.py <sqlite_file> <output_csv> <nvtx_range_name>")
        sys.exit(1)

    db_path = sys.argv[1]
    output_csv = sys.argv[2]
    nvtx_range_name = sys.argv[3]

    export_faults_for_nvtx_range(db_path, output_csv, nvtx_range_name)
