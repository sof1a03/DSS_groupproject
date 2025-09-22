-- Purpose: Example of a time-series table (e.g., tourist tax from 84120NED)
-- Why: Demonstrates handling CBS period codes and storing a parsed date.
-- The ETL (jobs/cbs_time_series.py) parses 'Perioden' into year/frequency/date.

CREATE TABLE IF NOT EXISTS cbs.tourist_tax (
  periodcode    TEXT,         -- original CBS period code (e.g., '2019', '2019KW01', '2019MM03')
  value       NUMERIC,      -- numeric value
  year        INT,          -- derived from 'periodcode'
  frequency   TEXT,         -- 'Y' (year), 'Q' (quarter), 'M' (month)
  count       INT,          -- quarter or month index (1..4 or 1..12)
  date        DATE,         -- parsed first day of the period (useful for charts)
  inserted_at TIMESTAMPTZ DEFAULT now()
);

-- Speed up time-based filtering
CREATE INDEX IF NOT EXISTS idx_tourist_tax_date ON cbs.tourist_tax (date);
