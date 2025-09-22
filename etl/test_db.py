import psycopg2

DDL = """
CREATE SCHEMA IF NOT EXISTS core;
CREATE TABLE IF NOT EXISTS core.hello_etl (
    id SERIAL PRIMARY KEY,
    note TEXT NOT NULL
);
"""

def smoke_test(conn_info: dict):
    dsn = "host={host} port={port} dbname={dbname} user={user} password={password}".format(**conn_info)
    with psycopg2.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute("INSERT INTO core.hello_etl (note) VALUES (%s)", ("hi from etl",))
            cur.execute("SELECT id, note FROM core.hello_etl ORDER BY id DESC LIMIT 1;")
            row = cur.fetchone()
            print("[ETL] Read back:", row)
