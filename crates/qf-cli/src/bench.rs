//! A benchmark that measures the optimiser rather than asserting it.
//!
//! Every query is run twice — once on the bound plan and once on the optimised
//! one — against the same data in the same process. The numbers in the README
//! come from here, and anyone can regenerate them with `queryforge bench`.

use qf_common::{DataType, Field, Result, Schema, Value};
use qf_exec::operator::{collect, Operator};
use qf_exec::Session;
use qf_storage::array::ArrayBuilder;
use qf_storage::batch::{RecordBatch, DEFAULT_BATCH_SIZE};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Rows in the generated table when no size is given.
pub const DEFAULT_ROWS: usize = 500_000;
/// Rows per row group. Smaller groups prune more finely.
pub const BENCH_ROW_GROUP_SIZE: usize = 8_192;

pub struct Case {
    pub name: &'static str,
    pub sql: &'static str,
}

pub const CASES: &[Case] = &[
    Case {
        name: "selective range",
        sql: "SELECT id, amount FROM events WHERE id > 490000",
    },
    Case {
        name: "point lookup",
        sql: "SELECT label FROM events WHERE id = 123456",
    },
    Case {
        name: "narrow projection",
        sql: "SELECT sum(amount) FROM events",
    },
    Case {
        name: "group by",
        sql: "SELECT status, count(*), sum(amount) FROM events GROUP BY status",
    },
    Case {
        name: "filtered group by",
        sql: "SELECT status, count(*) FROM events WHERE amount > 900 GROUP BY status",
    },
    Case {
        name: "order by with limit",
        sql: "SELECT id, amount FROM events ORDER BY amount DESC LIMIT 10",
    },
    Case {
        name: "join",
        sql: "SELECT e.id, s.description FROM events e \
              JOIN statuses s ON e.status = s.name WHERE e.id > 495000",
    },
];

#[derive(Debug, Clone)]
pub struct Measurement {
    pub name: &'static str,
    pub optimized: Duration,
    pub unoptimized: Duration,
    pub rows: usize,
    pub row_groups_read: usize,
    pub row_groups_pruned: usize,
}

impl Measurement {
    pub fn speedup(&self) -> f64 {
        if self.optimized.as_secs_f64() == 0.0 {
            return f64::INFINITY;
        }
        self.unoptimized.as_secs_f64() / self.optimized.as_secs_f64()
    }
}

/// Builds the benchmark tables in `dir` and returns a session over them.
pub fn prepare(dir: &Path, rows: usize) -> Result<Session> {
    let mut session = Session::new();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("label", DataType::Utf8, false),
        Field::new("amount", DataType::Float64, true),
    ]));
    let statuses = ["pending", "shipped", "delivered", "cancelled"];

    let mut batches = Vec::new();
    let mut written = 0;
    while written < rows {
        let n = DEFAULT_BATCH_SIZE.min(rows - written);
        let mut ids = ArrayBuilder::new(DataType::Int64);
        let mut status = ArrayBuilder::new(DataType::Utf8);
        let mut label = ArrayBuilder::new(DataType::Utf8);
        let mut amount = ArrayBuilder::new(DataType::Float64);
        for i in written..written + n {
            ids.push(Value::Int64(i as i64))?;
            status.push(Value::Utf8(statuses[i % statuses.len()].to_string()))?;
            label.push(Value::Utf8(format!("event-{i}")))?;
            // Every hundredth row is NULL, so the null paths are exercised.
            if i % 100 == 0 {
                amount.push_null();
            } else {
                amount.push(Value::Float64((i % 1000) as f64))?;
            }
        }
        batches.push(RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                ids.finish()?,
                status.finish()?,
                label.finish()?,
                amount.finish()?,
            ],
        )?);
        written += n;
    }
    session.register("events", Arc::clone(&schema), batches)?;
    session.save_table_with_row_group_size(
        "events",
        dir.join("events.qfc"),
        BENCH_ROW_GROUP_SIZE,
    )?;

    let status_schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("description", DataType::Utf8, false),
    ]));
    let names: Vec<Value> = statuses
        .iter()
        .map(|s| Value::Utf8(s.to_string()))
        .collect();
    let descriptions: Vec<Value> = statuses
        .iter()
        .map(|s| Value::Utf8(format!("status: {s}")))
        .collect();
    session.register(
        "statuses",
        Arc::clone(&status_schema),
        vec![RecordBatch::try_new(
            status_schema,
            vec![
                qf_storage::array::Array::from_values(DataType::Utf8, &names)?,
                qf_storage::array::Array::from_values(DataType::Utf8, &descriptions)?,
            ],
        )?],
    )?;

    Ok(session)
}

fn time(session: &mut Session, sql: &str, optimized: bool) -> Result<(Duration, usize, Metrics)> {
    let mut op = session.operators_with(sql, optimized)?;
    let start = Instant::now();
    let batches = collect(op.as_mut())?;
    let elapsed = start.elapsed();
    let rows = batches.iter().map(RecordBatch::num_rows).sum();
    Ok((elapsed, rows, scan_metrics(op.as_ref())))
}

#[derive(Debug, Default, Clone, Copy)]
struct Metrics {
    read: usize,
    pruned: usize,
}

/// Sums the row-group counters of every scan in the tree.
fn scan_metrics(op: &dyn Operator) -> Metrics {
    let m = op.metrics();
    let mut total = Metrics {
        read: m.row_groups_read,
        pruned: m.row_groups_pruned,
    };
    for c in op.children() {
        let child = scan_metrics(c);
        total.read += child.read;
        total.pruned += child.pruned;
    }
    total
}

/// Runs every case, optimised and not.
pub fn run(session: &mut Session) -> Result<Vec<Measurement>> {
    let mut out = Vec::new();
    for case in CASES {
        // One untimed run first, so the comparison is not measuring the cost
        // of warming the file cache on whichever query happened to go first.
        time(session, case.sql, true)?;
        time(session, case.sql, false)?;

        let (optimized, rows, metrics) = time(session, case.sql, true)?;
        let (unoptimized, unopt_rows, _) = time(session, case.sql, false)?;
        // The two plans must agree, or the optimiser is not preserving
        // semantics and the timings mean nothing.
        if rows != unopt_rows {
            return Err(qf_common::Error::internal(format!(
                "`{}` returned {rows} rows optimised and {unopt_rows} unoptimised",
                case.name
            )));
        }
        out.push(Measurement {
            name: case.name,
            optimized,
            unoptimized,
            rows,
            row_groups_read: metrics.read,
            row_groups_pruned: metrics.pruned,
        });
    }
    Ok(out)
}

pub fn report(rows: usize, measurements: &[Measurement]) -> String {
    let mut out =
        format!("queryforge benchmark — {rows} rows, {BENCH_ROW_GROUP_SIZE}-row row groups\n\n");
    out.push_str(&format!(
        "{:<22} {:>10} {:>12} {:>9} {:>10} {:>14}\n",
        "query", "rows", "optimised", "speedup", "unopt.", "row groups"
    ));
    out.push_str(&"-".repeat(82));
    out.push('\n');
    for m in measurements {
        out.push_str(&format!(
            "{:<22} {:>10} {:>10.2}ms {:>8.1}x {:>8.2}ms {:>6} read {:>3} pruned\n",
            m.name,
            m.rows,
            m.optimized.as_secs_f64() * 1000.0,
            m.speedup(),
            m.unoptimized.as_secs_f64() * 1000.0,
            m.row_groups_read,
            m.row_groups_pruned,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new() -> TempDir {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "queryforge-bench-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn the_benchmark_data_is_built_and_queryable() {
        let dir = TempDir::new();
        let mut s = prepare(&dir.0, 20_000).unwrap();
        let b = s.query("SELECT count(*) FROM events").unwrap();
        assert_eq!(b.row(0)[0], Value::Int64(20_000));
        assert_eq!(
            s.query("SELECT count(*) FROM statuses").unwrap().num_rows(),
            1
        );
    }

    #[test]
    fn every_case_runs_and_both_plans_agree() {
        // If the optimiser changed an answer this would fail rather than
        // reporting a flattering speedup.
        let dir = TempDir::new();
        let mut s = prepare(&dir.0, 20_000).unwrap();
        let results = run(&mut s).unwrap();
        assert_eq!(results.len(), CASES.len());
        for m in &results {
            assert!(m.optimized.as_nanos() > 0, "{} was not timed", m.name);
        }
    }

    #[test]
    fn a_selective_predicate_prunes_row_groups() {
        let dir = TempDir::new();
        let mut s = prepare(&dir.0, 20_000).unwrap();
        let (_, rows, metrics) =
            time(&mut s, "SELECT id FROM events WHERE id > 19000", true).unwrap();
        assert_eq!(rows, 999);
        assert!(metrics.pruned > 0, "nothing was pruned");
        // Without the optimiser there is no pushed predicate, so nothing prunes.
        let (_, _, unopt) = time(&mut s, "SELECT id FROM events WHERE id > 19000", false).unwrap();
        assert_eq!(unopt.pruned, 0);
    }

    #[test]
    fn the_report_lists_every_case() {
        let m = vec![Measurement {
            name: "example",
            optimized: Duration::from_millis(1),
            unoptimized: Duration::from_millis(4),
            rows: 10,
            row_groups_read: 1,
            row_groups_pruned: 9,
        }];
        let text = report(1000, &m);
        assert!(text.contains("example"));
        assert!(text.contains("4.0x"));
        assert!(text.contains("9 pruned"));
        assert!(text.contains("1000 rows"));
    }

    #[test]
    fn a_zero_duration_reports_an_infinite_speedup_rather_than_dividing_by_zero() {
        let m = Measurement {
            name: "x",
            optimized: Duration::ZERO,
            unoptimized: Duration::from_millis(1),
            rows: 0,
            row_groups_read: 0,
            row_groups_pruned: 0,
        };
        assert!(m.speedup().is_infinite());
    }
}
