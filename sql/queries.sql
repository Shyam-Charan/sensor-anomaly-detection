-- Run with: python src/queries.py   (or open outputs/results.db in any SQLite client)

-- 1. Model comparison, best F1 first
SELECT model, ROUND(precision, 3) AS precision, ROUND(recall, 3) AS recall, ROUND(f1, 3) AS f1,
       alerts, windows_detected || '/' || windows_total AS windows_hit
FROM metrics
ORDER BY f1 DESC;

-- 2. Alerts per day for each model (JOIN-free aggregation with GROUP BY)
SELECT DATE(timestamp) AS day, model, SUM(predicted) AS alerts, SUM(label) AS labelled_anomaly_points
FROM predictions
GROUP BY day, model
HAVING alerts > 0
ORDER BY day, model;

-- 3. Days where a model alerted but nothing was labelled anomalous (candidate false alarms)
SELECT DATE(timestamp) AS day, model, SUM(predicted) AS alerts
FROM predictions
GROUP BY day, model
HAVING SUM(predicted) > 0 AND SUM(label) = 0
ORDER BY alerts DESC
LIMIT 10;

-- 4. Models that raised more alerts than the average model (subquery)
SELECT model, alerts FROM metrics
WHERE alerts > (SELECT AVG(alerts) FROM metrics);
