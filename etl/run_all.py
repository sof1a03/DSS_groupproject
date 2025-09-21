import os
from test_db import smoke_test

if __name__ == "__main__":
    conn_info = {
        "host": os.getenv("DB_HOST", "db_dashboard"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "dashboard"),
        "user": os.getenv("DB_USER", "student"),
        "password": os.getenv("DB_PASSWORD", "infomdss"),
    }
    print("[ETL] Starting smoke test with:", conn_info)
    smoke_test(conn_info)
    print("[ETL] Smoke test complete.")
