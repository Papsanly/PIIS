SELECT *
FROM flights
WHERE flights.late_aircraft_delay > (
    SELECT AVG(late_aircraft_delay)
    FROM flights
);
