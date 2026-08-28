//! The session: a catalog plus the ability to run statements against it.
//!
//! Everything the CLI does goes through here, and so does every end-to-end
//! test, which is deliberate — it means the tests exercise the same path a
//! user does rather than a convenient shortcut past it.

use crate::operator::{collect, Operator};
use crate::physical::{build, explain_analyze};
use qf_common::{Error, Result, Schema};
use qf_plan::binder::{bind, bind_constant_row, schema_from_columns};
use qf_plan::logical::LogicalPlan;
use qf_plan::optimizer::optimize;
use qf_sql::ast::{Query, Statement};
use qf_sql::Parser;
use qf_storage::batch::RecordBatch;
use qf_storage::catalog::{Catalog, Table, TableSource};
use qf_storage::csv::{read_csv_file, read_csv_with_schema};
use qf_storage::format::{write_file, DEFAULT_ROW_GROUP_SIZE};
use std::path::Path;
use std::sync::Arc;

/// What running one statement produced.
#[derive(Debug, Clone, PartialEq)]
pub enum Output {
    Rows {
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
    },
    /// A plan, from `EXPLAIN`.
    Plan(String),
    /// A confirmation, from a statement that does not return rows.
    Message(String),
}

impl Output {
    pub fn row_count(&self) -> usize {
        match self {
            Output::Rows { batches, .. } => batches.iter().map(RecordBatch::num_rows).sum(),
            _ => 0,
        }
    }
}

#[derive(Default)]
pub struct Session {
    catalog: Catalog,
}

impl Session {
    pub fn new() -> Session {
        Session::default()
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// Runs every statement in `sql`.
    pub fn execute(&mut self, sql: &str) -> Result<Vec<Output>> {
        Parser::parse_statements(sql)?
            .into_iter()
            .map(|s| self.run(s))
            .collect()
    }

    /// Runs exactly one statement.
    pub fn execute_one(&mut self, sql: &str) -> Result<Output> {
        let mut out = self.execute(sql)?;
        match out.len() {
            1 => Ok(out.remove(0)),
            n => Err(Error::plan(format!("expected one statement, found {n}"))),
        }
    }

    /// Runs a query and returns its rows as a single batch. The shape most
    /// tests want.
    pub fn query(&mut self, sql: &str) -> Result<RecordBatch> {
        match self.execute_one(sql)? {
            Output::Rows { schema, batches } => RecordBatch::concat(schema, &batches),
            other => Err(Error::plan(format!(
                "expected a query, but the statement produced {other:?}"
            ))),
        }
    }

    fn run(&mut self, statement: Statement) -> Result<Output> {
        match statement {
            Statement::Query(q) => self.run_query(&q),
            Statement::Explain { analyze, query } => self.run_explain(&query, analyze),
            Statement::CreateTable { name, columns } => {
                let schema = schema_from_columns(&columns)?;
                self.catalog
                    .register_batches(name.clone(), schema, vec![])?;
                Ok(Output::Message(format!("created table {name}")))
            }
            Statement::DropTable { name } => {
                self.catalog.drop_table(&name)?;
                Ok(Output::Message(format!("dropped table {name}")))
            }
            Statement::Insert { table, rows } => self.run_insert(&table, rows),
            Statement::Copy { table, path } => self.run_copy(&table, &path),
        }
    }

    fn plan_query(&self, query: &Query) -> Result<LogicalPlan> {
        optimize(bind(query, &self.catalog)?)
    }

    fn run_query(&mut self, query: &Query) -> Result<Output> {
        let plan = self.plan_query(query)?;
        let mut op = build(&plan, &self.catalog)?;
        let schema = op.schema();
        let batches = collect(op.as_mut())?;
        Ok(Output::Rows { schema, batches })
    }

    fn run_explain(&mut self, query: &Query, analyze: bool) -> Result<Output> {
        let plan = self.plan_query(query)?;
        if !analyze {
            return Ok(Output::Plan(plan.explain()));
        }
        // ANALYZE has to actually run the query — the counters are the point.
        let mut op = build(&plan, &self.catalog)?;
        let rows: usize = collect(op.as_mut())?
            .iter()
            .map(RecordBatch::num_rows)
            .sum();
        let mut text = explain_analyze(op.as_ref());
        text.push_str(&format!("\n{rows} rows returned\n"));
        Ok(Output::Plan(text))
    }

    fn run_insert(&mut self, table: &str, rows: Vec<Vec<qf_sql::ast::Expr>>) -> Result<Output> {
        let entry = self.catalog.get(table)?.clone();
        let TableSource::Memory(existing) = &entry.source else {
            return Err(Error::plan(format!(
                "`{table}` is backed by a file; INSERT only works on in-memory tables"
            )));
        };

        // The values go through the same binder and evaluator as everything
        // else — evaluated against one empty row — rather than having their
        // own literal-only path that could disagree with the rest of the
        // engine about types.
        let seed = RecordBatch::empty(Arc::new(Schema::empty()))?.take(&[0usize; 1])?;
        let mut builders: Vec<qf_storage::array::ArrayBuilder> = entry
            .schema
            .fields()
            .iter()
            .map(|f| qf_storage::array::ArrayBuilder::new(f.data_type))
            .collect();

        for row in &rows {
            if row.len() != entry.schema.len() {
                return Err(Error::plan(format!(
                    "`{table}` has {} columns but a row supplied {}",
                    entry.schema.len(),
                    row.len()
                )));
            }
            let bound = bind_constant_row(row, &self.catalog)?;
            for (i, e) in bound.iter().enumerate() {
                let column = crate::eval::evaluate(e, &seed)?;
                let target = entry.schema.field(i)?.data_type;
                builders[i].push(column.value(0).cast_to(target)?)?;
            }
        }

        let columns = builders
            .into_iter()
            .map(qf_storage::array::ArrayBuilder::finish)
            .collect::<Result<Vec<_>>>()?;
        let mut batches = existing.clone();
        batches.push(RecordBatch::try_new(Arc::clone(&entry.schema), columns)?);
        let inserted = rows.len();
        self.replace_batches(&entry.name, entry.schema.clone(), batches)?;
        Ok(Output::Message(format!("inserted {inserted} rows")))
    }

    fn run_copy(&mut self, table: &str, path: &str) -> Result<Output> {
        let entry = self.catalog.get(table)?.clone();
        let TableSource::Memory(existing) = &entry.source else {
            return Err(Error::plan(format!(
                "`{table}` is backed by a file; COPY only loads into in-memory tables"
            )));
        };
        let text = std::fs::read_to_string(path)?;
        let loaded = read_csv_with_schema(&text, Arc::clone(&entry.schema))?;
        let rows: usize = loaded.iter().map(RecordBatch::num_rows).sum();
        let mut batches = existing.clone();
        batches.extend(loaded);
        self.replace_batches(&entry.name, entry.schema.clone(), batches)?;
        Ok(Output::Message(format!("copied {rows} rows into {table}")))
    }

    fn replace_batches(
        &mut self,
        name: &str,
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
    ) -> Result<()> {
        self.catalog.drop_table(name)?;
        self.catalog.register_batches(name, schema, batches)
    }

    /// Creates a table from a CSV file, inferring its schema.
    pub fn import_csv(&mut self, name: &str, path: impl AsRef<Path>) -> Result<usize> {
        let (schema, batches) = read_csv_file(path, 1000)?;
        let rows = batches.iter().map(RecordBatch::num_rows).sum();
        if self.catalog.contains(name) {
            self.catalog.drop_table(name)?;
        }
        self.catalog.register_batches(name, schema, batches)?;
        Ok(rows)
    }

    /// Writes a table to a `.qfc` file and re-registers it as file-backed, so
    /// subsequent queries go through the columnar reader and its zone maps.
    pub fn save_table(&mut self, name: &str, path: impl AsRef<Path>) -> Result<usize> {
        self.save_table_with_row_group_size(name, path, DEFAULT_ROW_GROUP_SIZE)
    }

    pub fn save_table_with_row_group_size(
        &mut self,
        name: &str,
        path: impl AsRef<Path>,
        row_group_size: usize,
    ) -> Result<usize> {
        let entry = self.catalog.get(name)?.clone();
        let batches = match &entry.source {
            TableSource::Memory(b) => b.clone(),
            TableSource::File(p) => {
                let mut reader = qf_storage::format::QfcReader::open(p)?;
                reader.read_all()?
            }
        };
        let meta = write_file(
            path.as_ref(),
            Arc::clone(&entry.schema),
            &batches,
            row_group_size,
        )?;
        self.catalog.drop_table(name)?;
        self.catalog.register_file(&entry.name, path.as_ref())?;
        Ok(meta.num_rows())
    }

    /// Registers an existing `.qfc` file as a table.
    pub fn attach(&mut self, name: &str, path: impl AsRef<Path>) -> Result<usize> {
        if self.catalog.contains(name) {
            self.catalog.drop_table(name)?;
        }
        self.catalog.register_file(name, path)?;
        Ok(self.catalog.get(name)?.row_count())
    }

    /// Registers batches directly. Used by tests and by anything embedding the
    /// engine.
    pub fn register(
        &mut self,
        name: &str,
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
    ) -> Result<()> {
        if self.catalog.contains(name) {
            self.catalog.drop_table(name)?;
        }
        self.catalog.register_batches(name, schema, batches)
    }

    pub fn tables(&self) -> Vec<&Table> {
        self.catalog.tables().collect()
    }

    /// The optimised plan for a query, without running it.
    pub fn explain(&mut self, sql: &str) -> Result<String> {
        match Parser::parse_one(sql)? {
            Statement::Query(q) => Ok(self.plan_query(&q)?.explain()),
            other => Err(Error::plan(format!("cannot explain {other:?}"))),
        }
    }

    /// Builds the operator tree for a query without draining it. Lets a caller
    /// inspect per-operator counters after running it themselves.
    pub fn operators(&mut self, sql: &str) -> Result<Box<dyn Operator>> {
        self.operators_with(sql, true)
    }

    /// The same, with the optimiser optionally skipped.
    ///
    /// Running a query both ways is the only honest way to say what the
    /// optimiser is worth: the benchmark measures the difference rather than
    /// asserting it.
    pub fn operators_with(&mut self, sql: &str, optimized: bool) -> Result<Box<dyn Operator>> {
        match Parser::parse_one(sql)? {
            Statement::Query(q) => {
                let plan = bind(&q, &self.catalog)?;
                let plan = if optimized { optimize(plan)? } else { plan };
                build(&plan, &self.catalog)
            }
            other => Err(Error::plan(format!("cannot run {other:?}"))),
        }
    }
}
