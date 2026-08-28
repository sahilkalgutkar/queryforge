# The life of a query

One query, traced through all six stages. Everything below is real output from
`examples/tour.sql` and the shell.

```sql
SELECT c.name, count(*) AS orders
FROM orders o
JOIN customers c ON c.id = o.customer_id
WHERE o.amount > 200
GROUP BY c.name
ORDER BY orders DESC
LIMIT 5;
```

## 1. Lexing — `qf-sql/src/lexer.rs`

Text becomes tokens, each carrying the byte offset it started at so that a
later error can point at the text that caused it. Keywords are upper-cased so
nothing downstream needs a case-insensitive compare; identifiers keep the case
they were written in.

The cases that separate this from a toy tokeniser: `"quoted identifiers"` that
may hold keywords or spaces, doubled quotes as escapes inside both strings and
identifiers, `--` comments, exponent notation, and an integer too large for
`i64` falling back to a float rather than failing the query.

## 2. Parsing — `qf-sql/src/parser.rs`

Recursive descent for statements, precedence climbing for expressions. The tree
is purely *syntactic*: a column is a name the user typed, not a position, and a
function is a name, not an aggregate. That is what lets the next stage be the
only place that resolves anything.

Two precedence decisions are worth arguing about, and both have tests:

- `NOT` sits between `AND` and comparison, so `NOT a = 1` groups as
  `NOT (a = 1)` and `NOT a AND b` as `(NOT a) AND b`.
- `BETWEEN` parses its bounds at comparison precedence, so the `AND` separating
  them is not swallowed as a conjunction. `a BETWEEN 1 AND 5 AND b = 2` has to
  come out as `(a BETWEEN 1 AND 5) AND (b = 2)`.

## 3. Binding — `qf-plan/src/binder.rs`

Names become positions and every node learns its type. This is the only place in
the engine that consults the catalog, which is why every "no such column" and
"ambiguous column" message comes from one function.

What the binder decides here:

- **`c.id` and `o.customer_id` resolve to positions** in the joined pair. An
  unqualified name matching both tables is an error, not a first-match win.
- **The `ON` clause splits** into hash-join keys and a residual filter. Only an
  equality with one side reading purely left columns and the other purely right
  becomes a key — a hash join cannot build on anything else. Written `c.id =
  o.customer_id`, the sides are swapped so the left key really is the left
  input's.
- **`count(*)` is registered once** and referenced by both the projection and
  the `ORDER BY`, so it is computed once.
- **`ORDER BY orders`** resolves the alias back to the aggregate it names.
- **`c.name` must be in `GROUP BY`** — and it is. A bare column that is neither
  grouped nor aggregated is rejected rather than returning an arbitrary row.
- **`BETWEEN` and `CASE x WHEN …` desugar** into comparisons, so the optimiser
  and the evaluator never see them.

The result is a tree of `LogicalPlan` nodes, each of which can compute its own
output schema.

## 4. Optimising — `qf-plan/src/optimizer.rs`

Four rules, each of which must preserve the rows *and* the schema. The schema
half is what makes them composable — a rule may move a filter or reorder a
join, but the node above must not be able to tell.

```
qf> EXPLAIN SELECT id FROM orders WHERE amount > 200 AND region = 'eu';
Projection: id#0
  Scan: orders [id, region, amount], pushed: (amount#2 > 200) AND (region#1 = 'eu')
```

Note what is *not* in that plan: there is no `Filter` node. The predicate ended
up inside the scan, where a zone map can act on it, and the scan reads three of
five columns.

1. **Constant folding** — evaluate what is known, apply the boolean identities,
   drop dead `CASE` branches. It deliberately stops at division by zero and
   integer overflow: folding those would let the plan-time answer differ from
   the run-time one.
2. **Predicate pushdown** — split conjunctions so each part travels separately;
   rewrite through column-only projections; push each part into the side of the
   join it belongs to; end inside the scan. Never into the padded side of an
   outer join. A comma join with an equality in `WHERE` becomes a real inner
   join here rather than a cross product that gets filtered afterwards.
3. **Join reordering** — put the smallest relation at the bottom of an
   inner-join chain, preferring a relation that has a join key to what is
   already joined over a smaller unrelated one. Reordering permutes the output
   columns, so the rule wraps the result in a projection restoring the original
   order.
4. **Projection pushdown** — narrow every scan to the columns actually read, and
   renumber every expression above it. This is the part that has to be exactly
   right: getting it wrong reads the wrong column and reports a plausible wrong
   answer.

Cardinality estimates come from the zone maps the storage layer already wrote,
so none of this costs an extra pass over the data.

## 5. Physical planning — `qf-exec/src/physical.rs`

The logical plan says *what*; this decides *how*. Two choices are made here
because both depend on physical facts the logical plan does not model:

- **Which side of the join to build.** The build side is what has to fit in
  memory, so the smaller estimate wins — `customers` here. On an outer join the
  choice is forced instead: the side that may be NULL-padded has to be the one
  that can be scanned for non-matches at the end.
- **Whether the sort can be a top-k.** The `LIMIT 5` is pushed into the sort as
  a fetch hint, so it never holds more than five rows. Nothing about the logical
  plan changes; the sort just stops keeping rows it can prove it will discard.

## 6. Execution — `qf-exec/`

Operators pull batches from each other. Pulling is what lets a `LIMIT` stop a
scan early rather than running it to completion and discarding the rest.

- **Scan** walks row groups, asks each pushed predicate whether the group's zone
  map could satisfy it, skips the chunks entirely if not, decodes only the
  projected columns, and applies the predicates to what is left.
- **Hash join** collects `customers`, hashes it, and streams `orders` past it.
  A NULL key never joins.
- **Hash aggregate** builds a table from group key to accumulators in one pass.
  Groups come out in first-seen order, so results are reproducible run to run.
- **Sort** keeps the best five rows and nothing else.
- **Limit** takes them.

`EXPLAIN ANALYZE` reports what each operator actually did:

```
qf> EXPLAIN ANALYZE SELECT id FROM events WHERE id > 490000;
Projection (rows=9999, time=0.00ms)
  Scan (rows=9999, time=0.91ms, row groups=1 read, 7 pruned)

9999 rows returned
```

Seven of eight row groups were never read.

## Where each stage lives

| Stage | Crate | Knows about |
| --- | --- | --- |
| Lexing, parsing | `qf-sql` | SQL text. Not tables, not types, not memory |
| Binding, optimising | `qf-plan` | Schemas and statistics. Not how a batch is laid out |
| Physical planning, execution | `qf-exec` | Batches and operators. Never resolves a name |
| Storage | `qf-storage` | Bytes, encodings, zone maps. No idea SQL exists |

The dependency direction is one-way, and the split is the argument.
