//! Hash aggregation.
//!
//! One pass over the input builds a hash table from group key to a set of
//! accumulators, then the table is turned into output rows. Sorting the input
//! instead would be asymptotically worse and, for the grouped queries people
//! actually write, much slower.
//!
//! The accumulators are the interesting part: `count` and `sum` need one
//! number, `avg` needs two, and any of them under `DISTINCT` needs the set of
//! values seen — which is why `DISTINCT` is opt-in rather than the default.

use crate::eval::evaluate;
use crate::operator::{Metrics, Operator};
use qf_common::{DataType, Error, Field, Result, Schema, Value};
use qf_plan::expr::{AggFunc, BoundAggregate, BoundExpr};
use qf_storage::array::ArrayBuilder;
use qf_storage::batch::RecordBatch;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone)]
enum Accumulator {
    Count {
        n: i64,
        seen: Option<HashSet<Value>>,
    },
    Sum {
        total: Option<Value>,
        seen: Option<HashSet<Value>>,
    },
    MinMax {
        value: Value,
        is_min: bool,
    },
    Avg {
        total: f64,
        n: i64,
        seen: Option<HashSet<Value>>,
    },
}

impl Accumulator {
    fn new(agg: &BoundAggregate) -> Accumulator {
        let seen = agg.distinct.then(HashSet::new);
        match agg.func {
            AggFunc::Count => Accumulator::Count { n: 0, seen },
            AggFunc::Sum => Accumulator::Sum { total: None, seen },
            AggFunc::Min => Accumulator::MinMax {
                value: Value::Null,
                is_min: true,
            },
            AggFunc::Max => Accumulator::MinMax {
                value: Value::Null,
                is_min: false,
            },
            AggFunc::Avg => Accumulator::Avg {
                total: 0.0,
                n: 0,
                seen,
            },
        }
    }

    /// Folds one value in. `None` is the `count(*)` case, where there is no
    /// argument and every row counts.
    fn update(&mut self, value: Option<&Value>) -> Result<()> {
        // Every aggregate ignores NULL inputs. `count(*)` has no input at all,
        // so it counts the row regardless.
        let v = match value {
            None => {
                if let Accumulator::Count { n, .. } = self {
                    *n += 1;
                }
                return Ok(());
            }
            Some(v) if v.is_null() => return Ok(()),
            Some(v) => v,
        };

        let fresh = |seen: &mut Option<HashSet<Value>>| match seen {
            None => true,
            Some(s) => s.insert(v.clone()),
        };

        match self {
            Accumulator::Count { n, seen } => {
                if fresh(seen) {
                    *n += 1;
                }
            }
            Accumulator::Sum { total, seen } => {
                if !fresh(seen) {
                    return Ok(());
                }
                *total = Some(match total.take() {
                    None => v.clone(),
                    Some(t) => add(&t, v)?,
                });
            }
            Accumulator::MinMax { value: cur, is_min } => {
                *cur = if *is_min {
                    Value::min(cur, v)
                } else {
                    Value::max(cur, v)
                };
            }
            Accumulator::Avg { total, n, seen } => {
                if !fresh(seen) {
                    return Ok(());
                }
                let x = v
                    .as_f64()
                    .ok_or_else(|| Error::typ(format!("cannot average {}", v.type_name())))?;
                *total += x;
                *n += 1;
            }
        }
        Ok(())
    }

    fn finish(&self) -> Value {
        match self {
            Accumulator::Count { n, .. } => Value::Int64(*n),
            // `sum` over no rows is NULL, not zero — SQL distinguishes "no
            // rows" from "rows that summed to nothing".
            Accumulator::Sum { total, .. } => total.clone().unwrap_or(Value::Null),
            Accumulator::MinMax { value, .. } => value.clone(),
            Accumulator::Avg { total, n, .. } => {
                if *n == 0 {
                    Value::Null
                } else {
                    Value::Float64(total / *n as f64)
                }
            }
        }
    }
}

fn add(a: &Value, b: &Value) -> Result<Value> {
    Ok(match (a, b) {
        (Value::Int64(x), Value::Int64(y)) => Value::Int64(x.wrapping_add(*y)),
        _ => {
            let (Some(x), Some(y)) = (a.as_f64(), b.as_f64()) else {
                return Err(Error::typ(format!(
                    "cannot sum {} and {}",
                    a.type_name(),
                    b.type_name()
                )));
            };
            Value::Float64(x + y)
        }
    })
}

pub struct HashAggregateExec {
    input: Box<dyn Operator>,
    group_exprs: Vec<BoundExpr>,
    aggregates: Vec<BoundAggregate>,
    schema: Arc<Schema>,
    /// Insertion order is preserved so the output is deterministic run to run,
    /// which matters for tests and for anyone diffing two query results.
    order: Vec<Vec<Value>>,
    table: HashMap<Vec<Value>, usize>,
    accumulators: Vec<Vec<Accumulator>>,
    done: bool,
    metrics: Metrics,
}

impl HashAggregateExec {
    pub fn new(
        input: Box<dyn Operator>,
        group_exprs: Vec<(BoundExpr, String)>,
        aggregates: Vec<BoundAggregate>,
    ) -> HashAggregateExec {
        let mut fields: Vec<Field> = group_exprs
            .iter()
            .map(|(e, n)| Field::new(n, e.data_type(), true))
            .collect();
        fields.extend(
            aggregates
                .iter()
                .map(|a| Field::new(&a.output_name, a.data_type, true)),
        );
        HashAggregateExec {
            input,
            group_exprs: group_exprs.into_iter().map(|(e, _)| e).collect(),
            aggregates,
            schema: Arc::new(Schema::new(fields)),
            order: Vec::new(),
            table: HashMap::new(),
            accumulators: Vec::new(),
            done: false,
            metrics: Metrics::default(),
        }
    }

    fn consume(&mut self, batch: &RecordBatch) -> Result<()> {
        let keys = self
            .group_exprs
            .iter()
            .map(|e| evaluate(e, batch))
            .collect::<Result<Vec<_>>>()?;
        let args = self
            .aggregates
            .iter()
            .map(|a| match &a.arg {
                Some(e) => evaluate(e, batch).map(Some),
                None => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;

        for row in 0..batch.num_rows() {
            let key: Vec<Value> = keys.iter().map(|k| k.value(row)).collect();
            let slot = match self.table.get(&key) {
                Some(i) => *i,
                None => {
                    let i = self.accumulators.len();
                    self.accumulators
                        .push(self.aggregates.iter().map(Accumulator::new).collect());
                    self.table.insert(key.clone(), i);
                    self.order.push(key);
                    i
                }
            };
            for (a, arg) in self.accumulators[slot].iter_mut().zip(args.iter()) {
                let v = arg.as_ref().map(|c| c.value(row));
                a.update(v.as_ref())?;
            }
        }
        Ok(())
    }

    fn build_output(&mut self) -> Result<RecordBatch> {
        // A global aggregate over no rows still returns one row: count 0,
        // everything else NULL.
        if self.group_exprs.is_empty() && self.accumulators.is_empty() {
            self.accumulators
                .push(self.aggregates.iter().map(Accumulator::new).collect());
            self.order.push(vec![]);
        }

        let mut builders: Vec<ArrayBuilder> = self
            .schema
            .fields()
            .iter()
            .map(|f| ArrayBuilder::new(f.data_type))
            .collect();

        for (key, accs) in self.order.iter().zip(self.accumulators.iter()) {
            for (i, v) in key.iter().enumerate() {
                builders[i].push(v.clone())?;
            }
            for (i, a) in accs.iter().enumerate() {
                let target = self.group_exprs.len() + i;
                let value = a.finish();
                // `sum` over integers can overflow into a float column and
                // vice versa; the schema is the authority.
                let coerced = match (&value, builders[target].data_type()) {
                    (Value::Null, _) => Value::Null,
                    (v, DataType::Float64) if v.as_f64().is_some() => {
                        Value::Float64(v.as_f64().unwrap())
                    }
                    (v, _) => v.clone(),
                };
                builders[target].push(coerced)?;
            }
        }

        let columns = builders
            .into_iter()
            .map(ArrayBuilder::finish)
            .collect::<Result<Vec<_>>>()?;
        RecordBatch::try_new(Arc::clone(&self.schema), columns)
    }
}

impl Operator for HashAggregateExec {
    fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.done {
            return Ok(None);
        }
        let start = Instant::now();
        while let Some(b) = self.input.next_batch()? {
            self.consume(&b)?;
        }
        let out = self.build_output()?;
        self.done = true;
        self.metrics.elapsed += start.elapsed();
        self.metrics.record(&out);
        Ok(Some(out))
    }

    fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    fn children(&self) -> Vec<&dyn Operator> {
        vec![self.input.as_ref()]
    }

    fn name(&self) -> &'static str {
        "HashAggregate"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::collect_one;
    use crate::operators::ValuesExec;
    use qf_plan::expr::AggFunc;
    use qf_storage::array::Array;

    fn agg(func: AggFunc, arg: Option<BoundExpr>, distinct: bool, t: DataType) -> BoundAggregate {
        BoundAggregate {
            func,
            arg,
            distinct,
            output_name: func.name().to_string(),
            data_type: t,
        }
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
            Field::new("n", DataType::Int64, true),
        ]))
    }

    fn source(rows: &[(Option<&str>, Option<i64>)]) -> Box<dyn Operator> {
        let gs: Vec<Value> = rows
            .iter()
            .map(|(g, _)| g.map_or(Value::Null, |s| Value::Utf8(s.to_string())))
            .collect();
        let ns: Vec<Value> = rows
            .iter()
            .map(|(_, n)| n.map_or(Value::Null, Value::Int64))
            .collect();
        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Array::from_values(DataType::Utf8, &gs).unwrap(),
                Array::from_values(DataType::Int64, &ns).unwrap(),
            ],
        )
        .unwrap();
        Box::new(ValuesExec::new(schema(), vec![batch]))
    }

    fn n_col() -> BoundExpr {
        BoundExpr::column(1, "n", DataType::Int64)
    }

    fn run(rows: &[(Option<&str>, Option<i64>)], aggs: Vec<BoundAggregate>) -> RecordBatch {
        let mut op = HashAggregateExec::new(source(rows), vec![], aggs);
        collect_one(&mut op).unwrap()
    }

    #[test]
    fn sum_over_no_non_null_values_is_null_not_zero() {
        // SQL distinguishes "no rows to add up" from "rows adding to nothing".
        let out = run(
            &[(None, None), (None, None)],
            vec![agg(AggFunc::Sum, Some(n_col()), false, DataType::Int64)],
        );
        assert!(out.row(0)[0].is_null());
    }

    #[test]
    fn count_star_counts_null_rows_and_count_of_a_column_does_not() {
        let out = run(
            &[(None, Some(1)), (None, None)],
            vec![
                agg(AggFunc::Count, None, false, DataType::Int64),
                agg(AggFunc::Count, Some(n_col()), false, DataType::Int64),
            ],
        );
        assert_eq!(out.row(0)[0], Value::Int64(2));
        assert_eq!(out.row(0)[1], Value::Int64(1));
    }

    #[test]
    fn min_and_max_ignore_nulls_and_work_on_text() {
        let g = BoundExpr::column(0, "g", DataType::Utf8);
        let out = run(
            &[(Some("m"), None), (None, None), (Some("a"), None)],
            vec![
                agg(AggFunc::Min, Some(g.clone()), false, DataType::Utf8),
                agg(AggFunc::Max, Some(g), false, DataType::Utf8),
            ],
        );
        assert_eq!(out.row(0)[0], Value::Utf8("a".into()));
        assert_eq!(out.row(0)[1], Value::Utf8("m".into()));
    }

    #[test]
    fn avg_divides_by_the_non_null_count_and_is_null_over_nothing() {
        let out = run(
            &[(None, Some(1)), (None, Some(2)), (None, None)],
            vec![agg(AggFunc::Avg, Some(n_col()), false, DataType::Float64)],
        );
        assert_eq!(out.row(0)[0], Value::Float64(1.5));

        let empty = run(
            &[(None, None)],
            vec![agg(AggFunc::Avg, Some(n_col()), false, DataType::Float64)],
        );
        assert!(empty.row(0)[0].is_null());
    }

    #[test]
    fn distinct_folds_repeats_out_of_count_sum_and_avg() {
        let rows = [(None, Some(2)), (None, Some(2)), (None, Some(4))];
        let out = run(
            &rows,
            vec![
                agg(AggFunc::Count, Some(n_col()), true, DataType::Int64),
                agg(AggFunc::Sum, Some(n_col()), true, DataType::Int64),
                agg(AggFunc::Avg, Some(n_col()), true, DataType::Float64),
            ],
        );
        assert_eq!(out.row(0)[0], Value::Int64(2));
        assert_eq!(out.row(0)[1], Value::Int64(6));
        assert_eq!(out.row(0)[2], Value::Float64(3.0));
    }

    #[test]
    fn a_global_aggregate_over_an_empty_input_still_returns_one_row() {
        let mut op = HashAggregateExec::new(
            Box::new(ValuesExec::new(schema(), vec![])),
            vec![],
            vec![agg(AggFunc::Count, None, false, DataType::Int64)],
        );
        let out = collect_one(&mut op).unwrap();
        assert_eq!(out.num_rows(), 1);
        assert_eq!(out.row(0)[0], Value::Int64(0));
    }

    #[test]
    fn a_grouped_aggregate_over_an_empty_input_returns_no_rows() {
        let mut op = HashAggregateExec::new(
            Box::new(ValuesExec::new(schema(), vec![])),
            vec![(BoundExpr::column(0, "g", DataType::Utf8), "g".into())],
            vec![agg(AggFunc::Count, None, false, DataType::Int64)],
        );
        assert_eq!(collect_one(&mut op).unwrap().num_rows(), 0);
    }

    #[test]
    fn groups_come_back_in_the_order_they_were_first_seen() {
        let mut op = HashAggregateExec::new(
            source(&[
                (Some("z"), Some(1)),
                (Some("a"), Some(1)),
                (Some("z"), Some(1)),
            ]),
            vec![(BoundExpr::column(0, "g", DataType::Utf8), "g".into())],
            vec![agg(AggFunc::Count, None, false, DataType::Int64)],
        );
        let out = collect_one(&mut op).unwrap();
        assert_eq!(out.row(0)[0], Value::Utf8("z".into()));
        assert_eq!(out.row(0)[1], Value::Int64(2));
        assert_eq!(out.row(1)[0], Value::Utf8("a".into()));
        assert_eq!(op.name(), "HashAggregate");
        assert_eq!(op.children().len(), 1);
    }

    #[test]
    fn averaging_text_is_an_error_rather_than_a_wrong_number() {
        let g = BoundExpr::column(0, "g", DataType::Utf8);
        let mut op = HashAggregateExec::new(
            source(&[(Some("a"), None)]),
            vec![],
            vec![agg(AggFunc::Avg, Some(g), false, DataType::Float64)],
        );
        assert!(collect_one(&mut op).is_err());
    }

    #[test]
    fn summing_text_is_an_error() {
        let g = BoundExpr::column(0, "g", DataType::Utf8);
        let mut op = HashAggregateExec::new(
            source(&[(Some("a"), None), (Some("b"), None)]),
            vec![],
            vec![agg(AggFunc::Sum, Some(g), false, DataType::Int64)],
        );
        assert!(collect_one(&mut op).is_err());
    }

    #[test]
    fn a_sum_declared_as_a_float_column_widens_its_integer_result() {
        let out = run(
            &[(None, Some(3))],
            vec![agg(AggFunc::Sum, Some(n_col()), false, DataType::Float64)],
        );
        assert_eq!(out.row(0)[0], Value::Float64(3.0));
    }
}
