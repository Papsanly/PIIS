SELECT
    a.city AS City,
    COUNT(*) AS FlightCount
FROM
    flights f
JOIN
    airports a ON f.dest = a.iata_code
GROUP BY
    a.city
ORDER BY
    FlightCount DESC;
