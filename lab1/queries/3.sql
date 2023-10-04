WITH CityDelays AS (
    SELECT
        a.city AS City,
        COALESCE(SUM(f.late_aircraft_delay), 0) AS TotalDelay
    FROM
        flights f
    JOIN
        airports a ON f.dest = a.iata_code
    GROUP BY
        a.city
)

SELECT
    City,
    TotalDelay,
    'Minimum' AS DelayType
FROM
    CityDelays
WHERE
    TotalDelay = (
        SELECT
            COALESCE(MIN(TotalDelay), 0)
        FROM
            CityDelays
    )

UNION ALL

SELECT
    City,
    TotalDelay,
    'Maximum' AS DelayType
FROM
    CityDelays
WHERE
    TotalDelay = (
        SELECT
            COALESCE(MAX(TotalDelay), 0)
        FROM
            CityDelays
    );
