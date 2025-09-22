import os
from test_db import smoke_test
from jobs import cbs_wb, cbs_time_series, pdok_geometries

if __name__ == "__main__":
    conn_info = {
        "host": os.getenv("DB_HOST", "db_dashboard"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "dashboard"),
        "user": os.getenv("DB_USER", "student"),
        "password": os.getenv("DB_PASSWORD", "infomdss"),
    }

    print("[ETL] Smoke test start")
    smoke_test(conn_info)
    print("[ETL] Smoke test OK")

    print("[ETL] CBS Wijken & Buurten stats …")
    cbs_wb.run()
    print("[ETL] CBS Wijken & Buurten stats OK")

    print("[ETL] CBS Time series …")
    cbs_time_series.run()
    print("[ETL] CBS Time series OK")

    print("[ETL] PDOK Geometries …")
    pdok_geometries.run()
    print("[ETL] PDOK Geometries OK")
