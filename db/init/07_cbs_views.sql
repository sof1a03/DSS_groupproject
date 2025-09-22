CREATE OR REPLACE VIEW cbs.vw_gemeente_measure_latest AS
WITH latest AS (
  SELECT
    o."WijkenEnBuurten",
    o."MeasureName",
    o."Value",
    o."Perioden",
    ROW_NUMBER() OVER (
      PARTITION BY o."WijkenEnBuurten", o."MeasureName"
      ORDER BY o.inserted_at DESC
    ) AS rn
  FROM cbs.wb_observations o
)
SELECT
  g.statcode,
  g.statnaam,
  l."MeasureName" AS measurename,
  l."Value"       AS value,
  l."Perioden"    AS perioden,
  g.geometry
FROM latest l
JOIN cbs.gemeente_2017 g
  ON g.statcode = l."WijkenEnBuurten"
WHERE l.rn = 1;
