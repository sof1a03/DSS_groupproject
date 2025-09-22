-- Municipality geometries (matches ETL + view)
CREATE TABLE IF NOT EXISTS cbs.gemeente_2017 (
  statcode  TEXT PRIMARY KEY,
  statnaam  TEXT,
  geometry  geometry(MULTIPOLYGON, 4326)
);
CREATE INDEX IF NOT EXISTS idx_gemeente_2017_geom
  ON cbs.gemeente_2017 USING GIST (geometry);
