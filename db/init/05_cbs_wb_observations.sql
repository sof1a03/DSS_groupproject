CREATE TABLE IF NOT EXISTS cbs.wb_observations (
  "WijkenEnBuurten" TEXT,
  "Perioden"        TEXT,
  "MeasureName"     TEXT,
  "Value"           NUMERIC,
  inserted_at       TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_wb_obs_region  ON cbs.wb_observations ("WijkenEnBuurten");
CREATE INDEX IF NOT EXISTS idx_wb_obs_period  ON cbs.wb_observations ("Perioden");
CREATE INDEX IF NOT EXISTS idx_wb_obs_measure ON cbs.wb_observations ("MeasureName");

