-- A short tour of what the engine does. Run it with:
--   cargo run --release -- examples/tour.sql

\import orders examples/orders.csv
\import customers examples/customers.csv

-- Types are inferred from the file; `amount` is nullable because row 4 is blank.
\d orders

-- A plain filter and projection.
SELECT id, region, amount FROM orders WHERE amount > 200 ORDER BY amount DESC;

-- NULL is not a value: it satisfies neither `> 200` nor `<= 200`.
SELECT
  count(*)            AS rows,
  count(amount)       AS with_amount,
  sum(amount)         AS total,
  avg(amount)         AS mean
FROM orders;

-- Grouping, with a filter on the groups rather than the rows.
SELECT region, count(*) AS orders, sum(amount) AS revenue
FROM orders
GROUP BY region
HAVING count(*) > 2
ORDER BY revenue DESC;

-- A join, with the join key written the "wrong" way round on purpose —
-- the binder normalises it.
SELECT c.name, c.tier, count(*) AS orders, sum(o.amount) AS spend
FROM orders o
JOIN customers c ON c.id = o.customer_id
GROUP BY c.name, c.tier
ORDER BY spend DESC NULLS LAST;

-- A left join keeps customers who have never ordered anything.
SELECT c.name, o.id
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.id AND o.status = 'cancelled'
ORDER BY c.name;

-- The predicate ends up inside the scan rather than above it.
EXPLAIN SELECT id FROM orders WHERE amount > 200 AND region = 'eu';

-- Writing the table to the columnar format, then querying it again. The plan
-- is the same; the scan now skips row groups whose zone maps rule the
-- predicate out.
\save orders /tmp/queryforge-orders.qfc
EXPLAIN ANALYZE SELECT id FROM orders WHERE amount > 200;
