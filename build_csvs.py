import sqlite3
import pandas as pd
import os

# ---- CONFIG ----
db_path = "nsysReport__ibm-granite-granite-3.0-2b-base__steps__gradient_checkpointing__no_warmup__logging__optimiser_offload__activation_prefetch__4__3000.sqlite"
output_dir = "migration_csvs__4_3000_optimiser_offload_activation_prefetch"

# Migration cause mapping
migration_causes = {
    0: "UNKNOWN",
    1: "USER_PREFETCH",
    2: "PAGE_FAULT",
    3: "SPECULATIVE_PREFETCH",
    4: "EVICTION",
    5: "ACCESS_COUNTERS"
}

# ----------------

os.makedirs(output_dir, exist_ok=True)

conn = sqlite3.connect(db_path)

for cause_id, cause_name in migration_causes.items():
    query = f"""
        SELECT *
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE migrationCause = {cause_id}
    """

    df = pd.read_sql_query(query, conn)

    if not df.empty:
        output_file = os.path.join(
            output_dir, f"migration_{cause_id}_{cause_name}.csv"
        )
        df.to_csv(output_file, index=False)
        print(f"Created: {output_file} ({len(df)} rows)")
    else:
        print(f"No rows for {cause_name}")

conn.close()
