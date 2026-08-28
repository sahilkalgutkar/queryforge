# The `.qfc` file format

This is the format queryforge reads and writes. It exists so that predicate and
projection pushdown can be physical rather than cosmetic: a reader should be
able to decide what it needs from the metadata alone and then touch only those
bytes.

## Layout

```
┌────────┬──────────────────────────┬─────────┬────────────┬────────┐
│ "QFC1" │ row group 0..n           │ footer  │ footer len │ "QFC1" │
│  4 B   │ column chunks, back to   │ metadata│   u32 LE   │  4 B   │
│        │ back, one per column     │         │            │        │
└────────┴──────────────────────────┴─────────┴────────────┴────────┘
```

The footer is at the *end*, and its length is the last field before the trailing
magic. A reader seeks to `len - 8`, reads eight bytes, learns where the metadata
starts, and reads it in one more seek — without touching any column data. The
leading magic is checked too, so a file that is not a `.qfc` fails immediately
rather than producing nonsense.

Every integer that is not fixed-width is a LEB128 varint, zigzagged when signed.
Most of what the format writes is small — run lengths, dictionary indices,
string lengths — and a varint spends one byte on those instead of eight.

## Footer

```
uvarint  field_count
  for each field:
    bytes   name
    u8      type tag        0=BOOLEAN 1=INT64 2=FLOAT64 3=UTF8
    u8      nullable
uvarint  row_group_count
  for each row group:
    uvarint  num_rows
    uvarint  column_count       (must equal field_count)
    for each column chunk:
      u64      offset           absolute, from the start of the file
      u64      length
      u8       encoding tag     0=plain 1=dictionary 2=RLE
      <column statistics>
```

A footer whose column count disagrees with its schema is rejected rather than
read partially — a truncated or corrupt file is an expected failure mode, not a
bug, and every accessor in the reader is bounds-checked.

### Column statistics — the zone map

```
uvarint  null_count
uvarint  row_count
uvarint  distinct_count
<value>  min
<value>  max
```

A value is a one-byte tag followed by its payload: `0` NULL, `1` boolean (one
byte), `2` int64 (zigzag varint), `3` float64 (8 bytes LE), `4` string (length
then bytes).

`min`, `max`, `row_count` and `null_count` are exact. `distinct_count` is exact
per row group and an **upper bound** once merged across row groups, because two
groups can hold overlapping values and the footer cannot say which. The planner
uses it only to order joins, so an over-estimate costs a plan rather than an
answer; the code says so where it merges.

This is the structure that makes pruning work. Before reading a row group the
scan asks each pushed predicate whether the group's `[min, max]` could possibly
satisfy it. The answer is only ever used to *skip* work, so it errs toward
reading: anything undecidable comes back "may match".

| Predicate | Prunes when |
| --- | --- |
| `col = v` | `v` falls outside `[min, max]` |
| `col < v` | `min >= v` |
| `col <= v` | `min > v` |
| `col > v` | `max <= v` |
| `col >= v` | `max < v` |
| `col <> v` | the group is a single value equal to `v`, with no nulls |
| anything, with `v` NULL | always — a comparison with NULL is never true |
| anything, group all NULL | always |

A conjunction prunes when *either* half rules the group out; a disjunction only
when *both* do.

## Column chunks

```
u8       encoding tag
uvarint  num_rows
u8       has_validity
  if 1:  uvarint word_count, then word_count × u64 LE
<payload>                       only the non-null values
```

Only non-null values are written. A column that is 90% null costs a validity
bitmap and almost nothing else; a column with no nulls at all carries no bitmap.

### Plain

Values back to back: booleans bit-packed, int64 as zigzag varints, float64 as
8 bytes little-endian, strings as length-prefixed bytes.

### Dictionary

```
uvarint  dict_len
         dict_len values, plain-encoded
         one uvarint index per non-null row
```

A dictionary code pointing past the end of its dictionary is rejected.

### Run-length

```
uvarint  run_count
  for each run:
    <value>  plain-encoded
    uvarint  run length
```

A run that would describe more rows than the header declares is rejected, and so
is a chunk whose runs describe fewer rows than the validity mask expects — a
short chunk must fail rather than be silently padded.

### Choosing between them

The encoding is chosen per chunk, not per column, because the right answer
changes with the data: an `order_status` column can hold two distinct values in
one row group and forty in the next. Two cheap signals decide it, measured on
the chunk itself:

- average run length ≥ 4 → **RLE**
- otherwise, distinct/rows ≤ 0.5 → **dictionary**
- otherwise → **plain**

Chunks under 16 rows skip the analysis entirely. Both thresholds are
conservative, because each encoding adds a decode step and a chunk that would
barely benefit is better off plain.

On the test fixtures: RLE is more than 10x smaller than plain on a sorted
column, and a dictionary is 4x smaller than plain on a shuffled four-value
string column. Those ratios are asserted in the test suite, not estimated.

## Row groups

The default is 65,536 rows. This is the granularity of pruning, and it is a
trade: smaller groups prune more precisely but multiply the footer, and past a
point the metadata costs more than the reads it saves. The benchmark writes
8,192-row groups so pruning is visible at a size that fits on a page.

Batches handed to the writer are repacked into even row groups regardless of how
they arrived, so the file's layout does not depend on how the data happened to
be batched upstream.
