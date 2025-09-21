CREATE SCHEMA IF NOT EXISTS cbs;

CREATE TABLE IF NOT EXISTS cbs.grid100 (
  id TEXT PRIMARY KEY,
  geom geometry(POLYGON, 4326),
  raw jsonb,
  inserted_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_grid100_geom ON cbs.grid100 USING GIST (geom);
