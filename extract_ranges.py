import sqlite3
import csv
import sys
import re


def export_nvtx_ranges(db_path, output_csv):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    SELECT start, end, text
    FROM NVTX_EVENTS
    """

    cursor.execute(query)

    # pattern: step_<number>_<name>_(forward|backward)
    pattern = re.compile(r"step_\d+_.+_(forward|backward)")

    with open(output_csv, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["range_name", "start_ns", "end_ns"])

        for start, end, text in cursor.fetchall():

            if text is None:
                continue

            if pattern.match(text):
                writer.writerow([text, start, end])

    conn.close()


if __name__ == "__main__":

    db_path = sys.argv[1]
    output_csv = sys.argv[2]

    export_nvtx_ranges(db_path, output_csv)
