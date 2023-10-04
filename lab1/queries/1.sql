SELECT
    a.city AS City,
    SUM(f.late_aircraft_delay) AS TotalDelay
FROM
    flights f
JOIN
    airports a ON f.dest = a.iata_code
GROUP BY
    a.city
ORDER BY
    TotalDelay DESC;
