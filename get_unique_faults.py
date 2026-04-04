import csv
import sys
from collections import defaultdict

def count_first_occurrences(csv_file):

    seen = set()
    access_counts = defaultdict(int)
    total_faults = 0
    act_total=0

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            addr = row["fault_address"]
            access = row["access_type"]
            faults = int(row["number of page faults"])
            act_total+=1

            if addr not in seen:
                seen.add(addr)

                total_faults += 1
                access_counts[access] += 1

    print("Unique first addresses:", len(seen))
    print("Total page faults (first occurrence only):", total_faults)
    print("total page faults sare ji sare: ",act_total)
    print("\nBreakdown by access_type:")
    for k,v in access_counts.items():
        print(f"access_type {k} : {v}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python first_fault_count.py faults.csv")
        sys.exit(1)

    count_first_occurrences(sys.argv[1])
