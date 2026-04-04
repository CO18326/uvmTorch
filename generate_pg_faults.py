import sqlite3
import csv
import sys

def export_page_faults_to_csv(db_path, output_csv):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT start, address, faultAccessType, numberOfPageFaults
        FROM CUDA_UM_GPU_PAGE_FAULT_EVENTS
        ORDER BY start;
    """

    cursor.execute(query)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "fault_address","access_type","number of page faults"])

        for row in cursor:
            timestamp_ns = row[0]
            fault_address = hex(row[1])
            access_type = row[2]
            num_pg_fault=row[3]
                                    # convert to hex for readability
            writer.writerow([timestamp_ns, fault_address,access_type,num_pg_fault])

    conn.close()
    print(f"CSV written to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python export_faults.py <sqlite_file> <output_csv>")
        sys.exit(1)

    db_path = sys.argv[1]
    output_csv = sys.argv[2]

    export_page_faults_to_csv(db_path, output_csv)
