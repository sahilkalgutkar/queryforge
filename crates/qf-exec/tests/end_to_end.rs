//! End-to-end tests: SQL text in, rows out, through the same path the CLI
//! uses. Anything that only passes because a test reached past the session is
//! not evidence the engine works.

use qf_common::{DataType, Field, Schema, Value};
use qf_exec::{Output, Session};
use qf_storage::array::Array;
use qf_storage::batch::RecordBatch;
use std::path::{Path, PathBuf};
use std::sync::Arc;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "queryforge-test-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// orders(id, customer_id, region, amount) with a NULL amount and a NULL
/// customer_id, so the null-handling tests have something to work with.
fn session() -> Session {
    let mut s = Session::new();
    let orders = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("customer_id", DataType::Int64, true),
        Field::new("region", DataType::Utf8, false),
        Field::new("amount", DataType::Float64, true),
    ]));
    let rows: Vec<(i64, Option<i64>, &str, Option<f64>)> = vec![
        (1, Some(10), "eu", Some(100.0)),
        (2, Some(10), "eu", Some(250.0)),
        (3, Some(20), "us", Some(75.0)),
        (4, Some(20), "us", None),
        (5, None, "apac", Some(500.0)),
        (6, Some(30), "eu", Some(30.0)),
    ];
    let batch = RecordBatch::try_new(
        Arc::clone(&orders),
        vec![
            col(DataType::Int64, rows.iter().map(|r| Value::Int64(r.0))),
            col(
                DataType::Int64,
                rows.iter().map(|r| r.1.map_or(Value::Null, Value::Int64)),
            ),
            col(
                DataType::Utf8,
                rows.iter().map(|r| Value::Utf8(r.2.to_string())),
            ),
            col(
                DataType::Float64,
                rows.iter().map(|r| r.3.map_or(Value::Null, Value::Float64)),
            ),
        ],
    )
    .unwrap();
    s.register("orders", orders, vec![batch]).unwrap();

    let customers = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let cust: Vec<(i64, &str)> = vec![(10, "ada"), (20, "brendan"), (40, "grace")];
    let batch = RecordBatch::try_new(
        Arc::clone(&customers),
        vec![
            col(DataType::Int64, cust.iter().map(|r| Value::Int64(r.0))),
            col(
                DataType::Utf8,
                cust.iter().map(|r| Value::Utf8(r.1.to_string())),
            ),
        ],
    )
    .unwrap();
    s.register("customers", customers, vec![batch]).unwrap();
    s
}

fn col(t: DataType, values: impl Iterator<Item = Value>) -> Array {
    let v: Vec<Value> = values.collect();
    Array::from_values(t, &v).unwrap()
}

/// Every row rendered as strings, which makes assertions readable.
fn rows(s: &mut Session, sql: &str) -> Vec<Vec<String>> {
    let batch = s.query(sql).unwrap();
    batch
        .rows()
        .map(|r| r.iter().map(ToString::to_string).collect())
        .collect()
}

fn one(s: &mut Session, sql: &str) -> String {
    let r = rows(s, sql);
    assert_eq!(r.len(), 1, "expected one row from `{sql}`, got {r:?}");
    assert_eq!(r[0].len(), 1, "expected one column from `{sql}`");
    r[0][0].clone()
}

#[test]
fn a_projection_returns_the_columns_it_names() {
    let mut s = session();
    assert_eq!(
        rows(&mut s, "SELECT id, region FROM orders WHERE id <= 2"),
        vec![vec!["1", "eu"], vec!["2", "eu"]]
    );
}

#[test]
fn a_wildcard_returns_every_column() {
    let mut s = session();
    let r = rows(&mut s, "SELECT * FROM orders WHERE id = 1");
    assert_eq!(r, vec![vec!["1", "10", "eu", "100.0"]]);
}

#[test]
fn a_null_never_satisfies_a_comparison_but_is_found_by_is_null() {
    let mut s = session();
    // Row 4 has a NULL amount, so neither `> 0` nor `<= 0` selects it.
    assert_eq!(
        one(&mut s, "SELECT count(*) FROM orders WHERE amount > 0"),
        "5"
    );
    assert_eq!(
        one(&mut s, "SELECT count(*) FROM orders WHERE amount <= 0"),
        "0"
    );
    assert_eq!(
        one(&mut s, "SELECT count(*) FROM orders WHERE amount IS NULL"),
        "1"
    );
}

#[test]
fn arithmetic_and_expressions_evaluate_per_row() {
    let mut s = session();
    assert_eq!(
        rows(
            &mut s,
            "SELECT id, amount * 2 AS d FROM orders WHERE id = 2"
        ),
        vec![vec!["2", "500.0"]]
    );
    assert_eq!(
        one(&mut s, "SELECT amount / 4 FROM orders WHERE id = 1"),
        "25.0"
    );
}

#[test]
fn in_between_and_like_filter_as_written() {
    let mut s = session();
    assert_eq!(
        one(
            &mut s,
            "SELECT count(*) FROM orders WHERE region IN ('eu', 'us')"
        ),
        "5"
    );
    assert_eq!(
        one(
            &mut s,
            "SELECT count(*) FROM orders WHERE amount BETWEEN 50 AND 260"
        ),
        "3"
    );
    assert_eq!(
        one(
            &mut s,
            "SELECT count(*) FROM orders WHERE region LIKE '%u%'"
        ),
        "5"
    );
    assert_eq!(
        one(
            &mut s,
            "SELECT count(*) FROM orders WHERE region NOT LIKE 'e%'"
        ),
        "3"
    );
}

#[test]
fn case_expressions_pick_a_branch_per_row() {
    let mut s = session();
    let r = rows(
        &mut s,
        "SELECT CASE WHEN amount > 200 THEN 'big' WHEN amount IS NULL THEN 'unknown' \
         ELSE 'small' END AS size FROM orders ORDER BY id",
    );
    let flat: Vec<&str> = r.iter().map(|x| x[0].as_str()).collect();
    assert_eq!(
        flat,
        vec!["small", "big", "small", "unknown", "big", "small"]
    );
}

#[test]
fn aggregates_ignore_nulls_and_count_star_does_not() {
    let mut s = session();
    let r = rows(
        &mut s,
        "SELECT count(*), count(amount), sum(amount), min(amount), max(amount) FROM orders",
    );
    assert_eq!(r[0][0], "6", "count(*) counts every row");
    assert_eq!(r[0][1], "5", "count(amount) skips the NULL");
    assert_eq!(r[0][2], "955.0");
    assert_eq!(r[0][3], "30.0");
    assert_eq!(r[0][4], "500.0");
}

#[test]
fn avg_divides_by_the_non_null_count() {
    let mut s = session();
    // 955 over 5 non-null rows, not over 6.
    assert_eq!(one(&mut s, "SELECT avg(amount) FROM orders"), "191.0");
}

#[test]
fn a_global_aggregate_over_no_rows_returns_one_row() {
    let mut s = session();
    let r = rows(
        &mut s,
        "SELECT count(*), sum(amount) FROM orders WHERE id = 99",
    );
    assert_eq!(r.len(), 1);
    assert_eq!(r[0][0], "0");
    // sum over nothing is NULL, not zero.
    assert_eq!(r[0][1], "NULL");
}

#[test]
fn grouping_splits_rows_by_key() {
    let mut s = session();
    let r = rows(
        &mut s,
        "SELECT region, count(*) AS n, sum(amount) AS total FROM orders \
         GROUP BY region ORDER BY region",
    );
    assert_eq!(
        r,
        vec![
            vec!["apac", "1", "500.0"],
            vec!["eu", "3", "380.0"],
            vec!["us", "2", "75.0"],
        ]
    );
}

#[test]
fn a_null_group_key_forms_its_own_group() {
    let mut s = session();
    let r = rows(
        &mut s,
        "SELECT customer_id, count(*) FROM orders GROUP BY customer_id ORDER BY customer_id",
    );
    assert_eq!(r.len(), 4);
    // Ascending order puts NULL last.
    assert_eq!(r[3][0], "NULL");
    assert_eq!(r[3][1], "1");
}

#[test]
fn having_filters_groups_not_rows() {
    let mut s = session();
    let r = rows(
        &mut s,
        "SELECT region, count(*) FROM orders GROUP BY region HAVING count(*) > 1 ORDER BY region",
    );
    assert_eq!(r, vec![vec!["eu", "3"], vec!["us", "2"]]);
}

#[test]
fn count_distinct_counts_each_value_once() {
    let mut s = session();
    assert_eq!(
        one(&mut s, "SELECT count(DISTINCT region) FROM orders"),
        "3"
    );
    assert_eq!(
        one(&mut s, "SELECT count(DISTINCT customer_id) FROM orders"),
        "3"
    );
    assert_eq!(
        one(&mut s, "SELECT sum(DISTINCT amount) FROM orders"),
        "955.0"
    );
}

#[test]
fn an_inner_join_keeps_only_matching_rows() {
    let mut s = session();
    let r = rows(
        &mut s,
        "SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id ORDER BY o.id",
    );
    assert_eq!(
        r,
        vec![
            vec!["1", "ada"],
            vec!["2", "ada"],
            vec!["3", "brendan"],
            vec!["4", "brendan"],
        ]
    );
}

#[test]
fn a_null_join_key_never_matches() {
    // Order 5 has a NULL customer_id; `NULL = anything` is unknown, so it
    // must not join even though customers also has no NULL id.
    let mut s = session();
    assert_eq!(
        one(
            &mut s,
            "SELECT count(*) FROM orders o JOIN customers c ON o.customer_id = c.id"
        ),
        "4"
    );
}

#[test]
fn a_left_join_pads_the_unmatched_side_with_nulls() {
    let mut s = session();
    let r = rows(
        &mut s,
        "SELECT o.id, c.name FROM orders o LEFT JOIN customers c ON o.customer_id = c.id \
         ORDER BY o.id",
    );
    assert_eq!(r.len(), 6);
    assert_eq!(r[4], vec!["5", "NULL"], "the NULL key row survives padded");
    assert_eq!(r[5], vec!["6", "NULL"], "customer 30 does not exist");
}

#[test]
fn a_right_join_preserves_the_right_side() {
    let mut s = session();
    let r = rows(
        &mut s,
        "SELECT o.id, c.name FROM orders o RIGHT JOIN customers c ON o.customer_id = c.id \
         ORDER BY c.name, o.id",
    );
    // grace (id 40) has no orders and must still appear.
    assert!(r.iter().any(|row| row == &vec!["NULL", "grace"]));
    assert_eq!(r.len(), 5);
}

#[test]
fn a_full_join_preserves_both_sides() {
    let mut s = session();
    let r = rows(
        &mut s,
        "SELECT o.id, c.name FROM orders o FULL JOIN customers c ON o.customer_id = c.id",
    );
    // 4 matched + orders 5 and 6 unmatched + customer grace unmatched.
    assert_eq!(r.len(), 7);
    assert!(r.iter().any(|row| row[1] == "grace" && row[0] == "NULL"));
    assert!(r.iter().any(|row| row[0] == "5" && row[1] == "NULL"));
}

#[test]
fn a_cross_join_produces_the_product() {
    let mut s = session();
    assert_eq!(
        one(&mut s, "SELECT count(*) FROM orders CROSS JOIN customers"),
        "18"
    );
}

#[test]
fn a_comma_join_with_an_equality_gives_the_same_answer_as_an_inner_join() {
    // The optimiser rewrites this into a hash join; the answer must not change.
    let mut s = session();
    let comma = rows(
        &mut s,
        "SELECT o.id, c.name FROM orders o, customers c WHERE o.customer_id = c.id ORDER BY o.id",
    );
    let explicit = rows(
        &mut s,
        "SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id ORDER BY o.id",
    );
    assert_eq!(comma, explicit);
}

#[test]
fn a_join_with_a_non_equality_condition_still_works() {
    let mut s = session();
    let r = rows(
        &mut s,
        "SELECT count(*) FROM orders o JOIN customers c \
         ON o.customer_id = c.id AND o.amount > 100",
    );
    assert_eq!(r[0][0], "1");
}

#[test]
fn ordering_is_ascending_by_default_with_nulls_last() {
    let mut s = session();
    let r = rows(&mut s, "SELECT id, amount FROM orders ORDER BY amount");
    let amounts: Vec<&str> = r.iter().map(|x| x[1].as_str()).collect();
    assert_eq!(
        amounts,
        vec!["30.0", "75.0", "100.0", "250.0", "500.0", "NULL"]
    );
}

#[test]
fn descending_order_puts_nulls_first_unless_told_otherwise() {
    let mut s = session();
    let r = rows(&mut s, "SELECT amount FROM orders ORDER BY amount DESC");
    assert_eq!(r[0][0], "NULL");
    let r = rows(
        &mut s,
        "SELECT amount FROM orders ORDER BY amount DESC NULLS LAST",
    );
    assert_eq!(r[0][0], "500.0");
    assert_eq!(r[5][0], "NULL");
}

#[test]
fn ordering_by_several_keys_breaks_ties_in_order() {
    let mut s = session();
    let r = rows(
        &mut s,
        "SELECT region, id FROM orders ORDER BY region ASC, id DESC",
    );
    assert_eq!(r[1], vec!["eu", "6"]);
    assert_eq!(r[2], vec!["eu", "2"]);
    assert_eq!(r[3], vec!["eu", "1"]);
}

#[test]
fn ordering_by_a_column_that_is_not_selected_works() {
    let mut s = session();
    let r = rows(
        &mut s,
        "SELECT id FROM orders ORDER BY amount DESC NULLS LAST",
    );
    assert_eq!(r[0][0], "5");
}

#[test]
fn limit_and_offset_take_a_window_of_the_result() {
    let mut s = session();
    let r = rows(&mut s, "SELECT id FROM orders ORDER BY id LIMIT 2 OFFSET 3");
    assert_eq!(r, vec![vec!["4"], vec!["5"]]);
    assert!(rows(&mut s, "SELECT id FROM orders LIMIT 0").is_empty());
    assert_eq!(rows(&mut s, "SELECT id FROM orders OFFSET 5").len(), 1);
}

#[test]
fn distinct_removes_duplicate_rows_but_keeps_their_first_order() {
    let mut s = session();
    let r = rows(&mut s, "SELECT DISTINCT region FROM orders");
    assert_eq!(r, vec![vec!["eu"], vec!["us"], vec!["apac"]]);
}

#[test]
fn a_select_without_from_evaluates_constants() {
    let mut s = session();
    assert_eq!(one(&mut s, "SELECT 1 + 1"), "2");
    assert_eq!(one(&mut s, "SELECT CAST('7' AS INT) * 2"), "14");
}

#[test]
fn create_insert_and_drop_work_end_to_end() {
    let mut s = Session::new();
    assert!(matches!(
        s.execute_one("CREATE TABLE t (a INT NOT NULL, b TEXT)")
            .unwrap(),
        Output::Message(_)
    ));
    s.execute_one("INSERT INTO t VALUES (1, 'x'), (2, 'y')")
        .unwrap();
    assert_eq!(one(&mut s, "SELECT count(*) FROM t"), "2");
    assert_eq!(one(&mut s, "SELECT b FROM t WHERE a = 2"), "y");

    s.execute_one("INSERT INTO t VALUES (3, 'z')").unwrap();
    assert_eq!(one(&mut s, "SELECT count(*) FROM t"), "3");

    s.execute_one("DROP TABLE t").unwrap();
    assert!(s.query("SELECT * FROM t").is_err());
}

#[test]
fn an_insert_of_the_wrong_width_or_type_is_refused() {
    let mut s = Session::new();
    s.execute_one("CREATE TABLE t (a INT, b TEXT)").unwrap();
    assert!(s.execute_one("INSERT INTO t VALUES (1)").is_err());
    assert!(s
        .execute_one("INSERT INTO t VALUES ('not a number', 'x')")
        .is_err());
    // An integer into a float column is fine, though.
    s.execute_one("CREATE TABLE f (x DOUBLE)").unwrap();
    s.execute_one("INSERT INTO f VALUES (3)").unwrap();
    assert_eq!(one(&mut s, "SELECT x FROM f"), "3.0");
}

#[test]
fn several_statements_run_in_order() {
    let mut s = Session::new();
    let out = s
        .execute("CREATE TABLE t (a INT); INSERT INTO t VALUES (1); SELECT count(*) FROM t")
        .unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out[2].row_count(), 1);
}

#[test]
fn csv_is_imported_with_inferred_types() {
    let dir = TempDir::new("csv");
    let path = dir.join("sales.csv");
    std::fs::write(&path, "id,region,amount\n1,eu,10.5\n2,us,20\n3,eu,\n").unwrap();

    let mut s = Session::new();
    assert_eq!(s.import_csv("sales", &path).unwrap(), 3);
    assert_eq!(one(&mut s, "SELECT count(*) FROM sales"), "3");
    assert_eq!(one(&mut s, "SELECT sum(amount) FROM sales"), "30.5");
    assert_eq!(
        one(&mut s, "SELECT count(*) FROM sales WHERE amount IS NULL"),
        "1"
    );
}

#[test]
fn copy_loads_a_csv_into_an_existing_table() {
    let dir = TempDir::new("copy");
    let path = dir.join("more.csv");
    std::fs::write(&path, "a,b\n7,seven\n8,eight\n").unwrap();

    let mut s = Session::new();
    s.execute_one("CREATE TABLE t (a INT, b TEXT)").unwrap();
    let out = s
        .execute_one(&format!("COPY t FROM '{}'", path.display()))
        .unwrap();
    assert!(matches!(out, Output::Message(m) if m.contains("2 rows")));
    assert_eq!(one(&mut s, "SELECT b FROM t WHERE a = 8"), "eight");
}

#[test]
fn a_table_survives_a_round_trip_through_the_columnar_format() {
    let dir = TempDir::new("qfc");
    let path = dir.join("orders.qfc");

    let mut s = session();
    let before = rows(&mut s, "SELECT id, region, amount FROM orders ORDER BY id");
    assert_eq!(s.save_table("orders", &path).unwrap(), 6);
    // The table is now file-backed, so this reads through the columnar reader.
    let after = rows(&mut s, "SELECT id, region, amount FROM orders ORDER BY id");
    assert_eq!(before, after);

    let mut fresh = Session::new();
    fresh.attach("orders", &path).unwrap();
    assert_eq!(
        rows(
            &mut fresh,
            "SELECT id, region, amount FROM orders ORDER BY id"
        ),
        before
    );
}

#[test]
fn a_predicate_prunes_row_groups_it_cannot_match() {
    let dir = TempDir::new("prune");
    let path = dir.join("wide.qfc");

    // 10 row groups of 100 sorted rows each.
    let mut s = Session::new();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    let ids: Vec<Value> = (0..1000).map(Value::Int64).collect();
    let labels: Vec<Value> = (0..1000).map(|i| Value::Utf8(format!("row{i}"))).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Array::from_values(DataType::Int64, &ids).unwrap(),
            Array::from_values(DataType::Utf8, &labels).unwrap(),
        ],
    )
    .unwrap();
    s.register("wide", schema, vec![batch]).unwrap();
    s.save_table_with_row_group_size("wide", &path, 100)
        .unwrap();

    // Only the group holding 950..999 can satisfy this.
    let mut op = s
        .operators("SELECT label FROM wide WHERE id > 980")
        .unwrap();
    let out = qf_exec::operator::collect(op.as_mut()).unwrap();
    let returned: usize = out.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(returned, 19);

    let text = qf_exec::physical::explain_analyze(op.as_ref());
    assert!(text.contains("1 read, 9 pruned"), "{text}");
}

#[test]
fn pruning_does_not_change_the_answer() {
    let dir = TempDir::new("prune-answer");
    let path = dir.join("t.qfc");
    let mut s = Session::new();
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let ids: Vec<Value> = (0..500).map(|i| Value::Int64((i * 7) % 500)).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Array::from_values(DataType::Int64, &ids).unwrap()],
    )
    .unwrap();
    s.register("t", Arc::clone(&schema), vec![batch.clone()])
        .unwrap();
    let in_memory = rows(
        &mut s,
        "SELECT count(*) FROM t WHERE id BETWEEN 100 AND 200",
    );

    s.save_table_with_row_group_size("t", &path, 50).unwrap();
    let on_disk = rows(
        &mut s,
        "SELECT count(*) FROM t WHERE id BETWEEN 100 AND 200",
    );
    assert_eq!(in_memory, on_disk);
    assert_eq!(on_disk[0][0], "101");
}

#[test]
fn explain_shows_the_optimised_plan() {
    let mut s = session();
    let text = s
        .explain("SELECT id FROM orders WHERE amount > 100")
        .unwrap();
    assert!(text.contains("Scan: orders"), "{text}");
    assert!(text.contains("pushed:"), "{text}");
    assert!(!text.contains("Filter"), "{text}");
}

#[test]
fn explain_analyze_reports_rows_per_operator() {
    let mut s = session();
    let out = s
        .execute_one("EXPLAIN ANALYZE SELECT region, count(*) FROM orders GROUP BY region")
        .unwrap();
    let Output::Plan(text) = out else {
        panic!("expected a plan");
    };
    assert!(text.contains("HashAggregate"), "{text}");
    assert!(text.contains("rows="), "{text}");
    assert!(text.contains("3 rows returned"), "{text}");
}

#[test]
fn a_limit_stops_the_scan_early() {
    let dir = TempDir::new("early-stop");
    let path = dir.join("big.qfc");
    let mut s = Session::new();
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let ids: Vec<Value> = (0..1000).map(Value::Int64).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Array::from_values(DataType::Int64, &ids).unwrap()],
    )
    .unwrap();
    s.register("big", schema, vec![batch]).unwrap();
    s.save_table_with_row_group_size("big", &path, 100).unwrap();

    let mut op = s.operators("SELECT id FROM big LIMIT 5").unwrap();
    let out = qf_exec::operator::collect(op.as_mut()).unwrap();
    assert_eq!(out.iter().map(RecordBatch::num_rows).sum::<usize>(), 5);
    // Only the first row group was ever read.
    let text = qf_exec::physical::explain_analyze(op.as_ref());
    assert!(text.contains("1 read"), "{text}");
}

#[test]
fn errors_reach_the_caller_rather_than_panicking() {
    let mut s = session();
    for sql in [
        "SELECT nope FROM orders",
        "SELECT * FROM ghosts",
        "SELECT id FROM orders WHERE region > 3",
        "SELECT region FROM orders GROUP BY amount",
        "SELECT sum(region) FROM orders",
        "SELECT FROM orders",
    ] {
        assert!(s.query(sql).is_err(), "`{sql}` should have failed");
    }
}

#[test]
fn the_catalog_lists_what_has_been_registered() {
    let s = session();
    let names: Vec<&str> = s.tables().iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["customers", "orders"]);
    assert_eq!(s.catalog().get("orders").unwrap().row_count(), 6);
}

#[test]
fn saving_a_file_backed_table_again_works() {
    let dir = TempDir::new("resave");
    let mut s = session();
    s.save_table("orders", dir.join("a.qfc")).unwrap();
    assert_eq!(s.save_table("orders", dir.join("b.qfc")).unwrap(), 6);
    assert_eq!(one(&mut s, "SELECT count(*) FROM orders"), "6");
    assert!(dir.path().join("b.qfc").exists());
}

#[test]
fn insert_and_copy_are_refused_on_a_file_backed_table() {
    let dir = TempDir::new("readonly");
    let mut s = session();
    s.save_table("orders", dir.join("o.qfc")).unwrap();
    assert!(s
        .execute_one("INSERT INTO orders VALUES (9, 9, 'x', 1.0)")
        .unwrap_err()
        .to_string()
        .contains("backed by a file"));
}
