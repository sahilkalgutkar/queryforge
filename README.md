# queryforge

[![CI](https://github.com/sahilkalgutkar/queryforge/actions/workflows/ci.yml/badge.svg)](https://github.com/sahilkalgutkar/queryforge/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sahilkalgutkar/queryforge/branch/main/graph/badge.svg)](https://codecov.io/gh/sahilkalgutkar/queryforge)
[![patch coverage](https://img.shields.io/badge/patch%20coverage-min%2080%25-blue.svg)](codecov.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust 1.82+](https://img.shields.io/badge/rust-1.82%2B-orange.svg)](https://www.rust-lang.org/)

I built queryforge to understand what actually happens between typing a SQL
query and getting rows back, by writing every stage of it in Rust: the lexer,
the parser, the binder, a cost-based optimiser, a vectorised execution engine,
and the columnar file format underneath all of it.

**Nothing is delegated.** There is no `sqlparser`, no DataFusion, no Arrow, no
Parquet — in fact no external crates at all. The whole workspace builds against
the standard library, which is deliberate: a query engine that wraps someone
else's parser and someone else's columnar format has skipped the two places
where the interesting decisions live.

```sql
qf> SELECT region, count(*) AS orders, sum(amount) AS revenue
    FROM orders GROUP BY region HAVING count(*) > 2 ORDER BY revenue DESC;
┌────────┬────────┬─────────┐
│ region │ orders │ revenue │
├────────┼────────┼─────────┤
│ us     │ 4      │ 615.25  │
│ eu     │ 4      │ 484.49  │
└────────┴────────┴─────────┘
2 rows
```

## Try it

```bash
cargo run --release -- examples/tour.sql
```

That walks the whole feature set on ten rows of orders: type inference from
CSV, filters and NULL semantics, aggregation, joins, `EXPLAIN`, and writing the
table out to the columnar format and querying it back.

Or start the shell:

```bash
cargo run --release
```

```
qf> \import orders examples/orders.csv
imported 10 rows into orders
qf> \d orders
orders (10 rows)
  id                   INT64      NOT NULL   min=1 max=10 nulls=0 distinct≈10
  customer_id          INT64      NOT NULL   min=10 max=50 nulls=0 distinct≈5
  region               UTF8       NOT NULL   min=apac max=us nulls=0 distinct≈3
  status               UTF8       NOT NULL   min=cancelled max=shipped nulls=0 distinct≈4
  amount               FLOAT64    NULL       min=45.25 max=510.0 nulls=1 distinct≈9
qf> EXPLAIN SELECT id FROM orders WHERE amount > 200 AND region = 'eu';
Projection: id#0
  Scan: orders [id, region, amount], pushed: (amount#2 > 200) AND (region#1 = 'eu')
```

The filter is not a node above the scan; it is *inside* it, which is what makes
the next section possible.

## What it does, and what that buys

### The columnar format

`.qfc` is my own. Column chunks are grouped into row groups; a footer at the end
records, for every chunk, its byte range, its encoding, and a zone map — the
min, max, null count and distinct count of the values it holds.

```
┌────────┬──────────────────────────┬─────────┬────────────┬────────┐
│ "QFC1" │ row group 0..n           │ footer  │ footer len │ "QFC1" │
│  4 B   │ column chunks, back to   │ metadata│   u32 LE   │  4 B   │
│        │ back, one per column     │         │            │        │
└────────┴──────────────────────────┴─────────┴────────────┴────────┘
```

The footer's length is the last field before the trailing magic, so a reader
seeks to `len - 8` and learns the entire layout without touching data. That is
what lets a query over two of forty columns read two chunks per row group, and
a query whose predicate falls outside a row group's `[min, max]` read none of
it at all.

Each chunk picks its own encoding by measuring the data rather than guessing
from its type, because the right answer changes between row groups of the same
column: long runs go RLE, low cardinality goes dictionary, everything else stays
plain. On the tests' own fixtures RLE is over 10x smaller than plain on a sorted
column, and a dictionary is 4x smaller on a shuffled four-value string column.

Strings use one contiguous byte buffer plus offsets rather than a `String` per
row, and nulls are a packed bitmap that a column without nulls does not carry
at all.

### The optimiser

Four rules, each of which must preserve both the rows a plan produces and the
schema it produces them in. The second half is what makes them composable, and
there is a test asserting it across a spread of queries.

| Rule | What it does |
| --- | --- |
| Constant folding | Evaluates what is already known, applies the boolean identities, drops dead `CASE` branches |
| Predicate pushdown | Splits conjunctions so each part travels separately, ending inside the scan |
| Join reordering | Puts the smallest relation at the bottom of an inner-join chain |
| Projection pushdown | Narrows every scan to the columns actually read |

Some decisions I'd defend in a review:

- **Folding stops at division by zero and integer overflow.** Folding those
  would let the plan-time answer differ from the run-time one, which is worse
  than not folding.
- **A predicate is never pushed into the padded side of an outer join.** Pushing
  `c.region = 3` into the right side of a `LEFT JOIN` deletes rows that should
  have come back NULL-padded. There is a test per join type.
- **A comma join with an equality in `WHERE` becomes a real inner join.**
  `FROM a, b WHERE a.id = b.id` is a hash join, not a cross product that gets
  filtered afterwards.
- **Join reordering prefers a relation that has a join key over a smaller
  unrelated one.** An accidental cross product is not something a later choice
  recovers from. Reordering permutes the output columns, so the rule wraps the
  result in a projection restoring the original order — the node above cannot
  tell.

### Execution

Operators pull batches from each other. That is not a style preference: it is
what lets `LIMIT 10` stop a scan after the first row group instead of running it
to completion and discarding the rest, and a test asserts exactly that.

Expression evaluation is vectorised — the operator is resolved once, outside the
loop, and the loop runs over a contiguous buffer. Comparing a column against a
literal has its own path so no constant column is materialised; same-type column
pairs skip boxing through a scalar type.

Sorting has three modes. Feeding a `LIMIT` it keeps only the best *k* rows, so
`ORDER BY ts DESC LIMIT 10` is bounded by 10 rather than by the table. Fitting in
memory it sorts once. Otherwise it writes sorted runs to spill files and merges
them — the only reason a database sorts differently from `Vec::sort`. A test runs
the same data through both paths and asserts identical output, because the answer
must not depend on whether the input fitted in memory.

The hash join builds one side and streams the other. For inner and cross joins
the smaller estimate is built, since the build side is what has to fit in memory.
For outer joins the choice is *forced*: the side that may be NULL-padded has to
be the one that can be scanned for non-matches at the end.

### NULL semantics

Three-valued logic is implemented properly, because getting it wrong silently
changes which rows come back rather than failing loudly:

- `false AND NULL` is `false` — the row cannot match whatever the unknown turns
  out to be. `true OR NULL` is `true`.
- A predicate that evaluates to unknown does not select its row.
- `5 IN (1, NULL)` is unknown, not false: the NULL might have been 5.
- A NULL join key never matches, including against another NULL.
- `sum` over no rows is NULL, not zero, and `count(*)` counts rows a
  `count(column)` would skip.
- `x / 0` yields NULL rather than aborting — one bad row should not kill a scan
  of a million.

## What the optimiser is worth

`queryforge bench` builds a table, runs every query twice — once on the bound
plan, once on the optimised one — and fails outright if the two disagree on the
row count. A flattering speedup from a wrong answer is worse than no number.

500,000 rows, 8192-row row groups, release build:

```
query                        rows    optimised   speedup     unopt.     row groups
----------------------------------------------------------------------------------
selective range              9999       0.48ms    154.6x    74.72ms      3 read  59 pruned
point lookup                    1       0.65ms    120.1x    78.47ms      1 read  61 pruned
narrow projection               1      18.73ms      4.5x    85.20ms     62 read   0 pruned
group by                        4      83.59ms      1.4x   119.67ms     62 read   0 pruned
filtered group by               4      41.94ms      1.9x    81.29ms     62 read   0 pruned
order by with limit            10      55.32ms      2.4x   131.88ms     62 read   0 pruned
join                         4999       1.22ms    123.6x   151.20ms      2 read  60 pruned
```

The three-figure numbers are zone-map pruning: reading 3 row groups instead of
62. The 4.5x on a full aggregate is projection pushdown on its own — one column
chunk per row group instead of four. Reproduce with `cargo run --release --
bench`.

`EXPLAIN ANALYZE` reports the same counters per operator:

```
qf> EXPLAIN ANALYZE SELECT id FROM events WHERE id > 490000;
Projection (rows=9999, time=0.00ms)
  Scan (rows=9999, time=0.91ms, row groups=1 read, 7 pruned)

9999 rows returned
```

(Eight row groups rather than sixty-two because `\save` uses the default
65,536-row groups; the benchmark writes smaller ones so pruning is finer.)

## Supported SQL

`SELECT [DISTINCT]` with `*`, `t.*` and aliases · `FROM` with inner, left,
right, full and cross joins · `WHERE` · `GROUP BY` (expressions or output
positions) · `HAVING` · `ORDER BY` with `ASC`/`DESC` and `NULLS FIRST|LAST` ·
`LIMIT`/`OFFSET`.

Expressions: arithmetic, comparison, `AND`/`OR`/`NOT`, `IS [NOT] NULL`,
`[NOT] BETWEEN`, `[NOT] IN`, `[NOT] LIKE` (with `%` and `_`), `CAST`, `CASE` in
both forms, and the aggregates `count`, `sum`, `min`, `max`, `avg` — each with
`DISTINCT`.

Statements: `CREATE TABLE`, `INSERT`, `COPY t FROM 'file.csv'`, `DROP TABLE`,
`EXPLAIN`, `EXPLAIN ANALYZE`.

**Not supported**, and I would rather say so than let you find out: subqueries,
CTEs, window functions, `UNION`, `UPDATE`/`DELETE`, transactions, indexes, and
concurrency. The type system is four types wide — `BOOLEAN`, `INT64`,
`FLOAT64`, `UTF8` — because every extra type multiplies out across the array
representations, the encodings, the comparison kernels and the accumulators, and
none of that would teach anything the existing four don't.

## How it's laid out

Six crates, split along the stages a query passes through:

| Crate | What lives there |
| --- | --- |
| `qf-common` | Types, values, schemas, and the error type tagged by the layer that raised it |
| `qf-storage` | Arrays, bitmaps, batches, encodings, the `.qfc` format, CSV ingest, the catalog |
| `qf-sql` | Lexer, syntax tree, parser |
| `qf-plan` | Binder, logical plan, cost model, optimiser |
| `qf-exec` | Vectorised evaluation, the operators, the physical planner, the session |
| `qf-cli` | Shell, result rendering, benchmark harness |

The dependency direction is one-way, and the split is the argument: `qf-sql` has
no idea what a table is, `qf-plan` has no idea how a batch is laid out in
memory, and `qf-exec` never resolves a name.

## Building and testing

```bash
cargo test --workspace       # 542 tests
cargo clippy --workspace --all-targets -- -D warnings
cargo run --release -- examples/tour.sql
cargo run --release -- bench --rows 500000
```

CI runs formatting, clippy with warnings denied, the full suite and coverage on
every push, and **fails the build if any test is ignored** — a test that quietly
stops running keeps a job green while covering nothing.

Coverage sits at roughly 97% of lines. The one exclusion is the REPL's terminal
loop, which is read-a-line-print-a-line plumbing; everything worth testing about
it lives in `shell.rs`, which the tests drive directly.

## Further reading

- [docs/file-format.md](docs/file-format.md) — the `.qfc` format byte by byte,
  the zone-map pruning rules, and how each chunk's encoding is chosen.
- [docs/query-lifecycle.md](docs/query-lifecycle.md) — one query traced through
  all six stages, with the real plan and counters at each step.

## Things I got wrong along the way

Worth recording, because they are the parts I actually learned from:

- **A `LEFT JOIN` with a non-equality in its `ON` clause dropped rows.** The
  condition was being applied *after* deciding which probe rows had matched, so
  a preserved row whose only candidate failed the condition vanished instead of
  coming back NULL-padded. Forty-three end-to-end tests missed it because every
  join test either had no residual condition or was an inner join. Running the
  example script found it in a minute.
- **A projection producing no columns dropped its row count**, which would have
  made `SELECT count(*)` return zero once column pruning removed every column.
- **A comment line above a statement swallowed it**, because the script reader
  buffered the comment and then read the next line as a continuation.
