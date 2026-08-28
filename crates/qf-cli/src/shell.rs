//! The shell's command handling, separated from the terminal loop so that
//! every command can be tested by feeding it a line and reading what it
//! returns — no pty, no stdin, no mocking.

use crate::render;
use qf_common::{Error, Result};
use qf_exec::{Output, Session};
use std::time::Instant;

pub struct Shell {
    session: Session,
    timing: bool,
}

impl Default for Shell {
    fn default() -> Self {
        Shell::new()
    }
}

impl Shell {
    pub fn new() -> Shell {
        Shell {
            session: Session::new(),
            timing: false,
        }
    }

    pub fn session(&mut self) -> &mut Session {
        &mut self.session
    }

    /// Runs one line of input — a backslash command or SQL — and returns what
    /// should be printed.
    ///
    /// Errors come back as `Err` rather than being printed here, so the caller
    /// decides whether a failure ends the session (a script) or just prints
    /// (the REPL).
    pub fn handle(&mut self, line: &str) -> Result<String> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(String::new());
        }
        // Comments are not stripped here: the lexer already skips them, and
        // treating a leading `--` as "ignore this input" would swallow the
        // statement that follows a comment on the line above it.
        if let Some(command) = line.strip_prefix('\\') {
            return self.meta(command);
        }
        let start = Instant::now();
        let outputs = self.session.execute(line)?;
        let mut text = String::new();
        for out in outputs {
            text.push_str(&self.render(out));
        }
        if self.timing {
            text.push_str(&format!(
                "time: {:.2}ms\n",
                start.elapsed().as_secs_f64() * 1000.0
            ));
        }
        Ok(text)
    }

    fn render(&self, out: Output) -> String {
        match out {
            Output::Rows { schema, batches } => render::table(&schema, &batches),
            Output::Plan(text) => text,
            Output::Message(m) => format!("{m}\n"),
        }
    }

    fn meta(&mut self, command: &str) -> Result<String> {
        let mut parts = command.split_whitespace();
        let name = parts.next().unwrap_or("");
        let args: Vec<&str> = parts.collect();

        match name {
            "?" | "h" | "help" => Ok(HELP.to_string()),
            "q" | "quit" => Ok(String::new()),
            "dt" => Ok(self.list_tables()),
            "d" => {
                let table = arg(&args, 0, "\\d <table>")?;
                self.describe(table)
            }
            "import" => {
                let (name, path) = two(&args, "\\import <table> <file.csv>")?;
                let rows = self.session.import_csv(name, path)?;
                Ok(format!("imported {rows} rows into {name}\n"))
            }
            "save" => {
                let (name, path) = two(&args, "\\save <table> <file.qfc>")?;
                let rows = self.session.save_table(name, path)?;
                Ok(format!(
                    "wrote {rows} rows to {path}; {name} now reads from it\n"
                ))
            }
            "attach" => {
                let (name, path) = two(&args, "\\attach <table> <file.qfc>")?;
                let rows = self.session.attach(name, path)?;
                Ok(format!("attached {name} ({rows} rows) from {path}\n"))
            }
            "timing" => {
                self.timing = !self.timing;
                Ok(format!(
                    "timing {}\n",
                    if self.timing { "on" } else { "off" }
                ))
            }
            "explain" => {
                let sql = command
                    .strip_prefix("explain")
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if sql.is_empty() {
                    return Err(Error::plan("usage: \\explain <query>".to_string()));
                }
                self.session.explain(&sql)
            }
            other => Err(Error::plan(format!(
                "unknown command `\\{other}` — try \\help"
            ))),
        }
    }

    fn list_tables(&self) -> String {
        let tables = self.session.tables();
        if tables.is_empty() {
            return "no tables — try \\import <name> <file.csv>\n".to_string();
        }
        let mut out = String::new();
        for t in tables {
            let kind = match &t.source {
                qf_storage::catalog::TableSource::Memory(_) => "memory",
                qf_storage::catalog::TableSource::File(p) => {
                    out.push_str(&format!(
                        "{:<20} {:>10} rows  file  {}\n",
                        t.name,
                        t.row_count(),
                        p.display()
                    ));
                    continue;
                }
            };
            out.push_str(&format!(
                "{:<20} {:>10} rows  {kind}\n",
                t.name,
                t.row_count()
            ));
        }
        out
    }

    fn describe(&self, table: &str) -> Result<String> {
        let t = self.session.catalog().get(table)?;
        let mut out = format!("{} ({} rows)\n", t.name, t.row_count());
        for (i, f) in t.schema.fields().iter().enumerate() {
            let null = if f.nullable { "NULL" } else { "NOT NULL" };
            let stats = match t.stats.column(i) {
                Some(s) if s.row_count > 0 => format!(
                    "  min={} max={} nulls={} distinct≈{}",
                    s.min, s.max, s.null_count, s.distinct_count
                ),
                _ => String::new(),
            };
            out.push_str(&format!(
                "  {:<20} {:<10} {null:<9}{stats}\n",
                f.name, f.data_type
            ));
        }
        Ok(out)
    }
}

fn arg<'a>(args: &[&'a str], i: usize, usage: &str) -> Result<&'a str> {
    args.get(i)
        .copied()
        .ok_or_else(|| Error::plan(format!("usage: {usage}")))
}

fn two<'a>(args: &[&'a str], usage: &str) -> Result<(&'a str, &'a str)> {
    Ok((arg(args, 0, usage)?, arg(args, 1, usage)?))
}

pub const HELP: &str = "\
queryforge — a columnar SQL engine

  SQL                       any supported statement, ending with an optional ;
  EXPLAIN <query>           show the optimised plan
  EXPLAIN ANALYZE <query>   run it and show per-operator counters

  \\dt                       list tables
  \\d <table>                show a table's columns and statistics
  \\import <t> <file.csv>    load a CSV into a new in-memory table
  \\save <t> <file.qfc>      write a table to the columnar format and read from it
  \\attach <t> <file.qfc>    register an existing .qfc file as a table
  \\explain <query>          same as EXPLAIN
  \\timing                   toggle query timing
  \\help                     this text
  \\q                        quit
";

#[cfg(test)]
mod tests {
    use super::*;

    fn shell() -> Shell {
        let mut s = Shell::new();
        s.handle("CREATE TABLE t (a INT, b TEXT)").unwrap();
        s.handle("INSERT INTO t VALUES (1, 'x'), (2, 'y')").unwrap();
        s
    }

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(tag: &str) -> TempDir {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "queryforge-shell-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
        fn file(&self, name: &str) -> String {
            self.0.join(name).to_string_lossy().into_owned()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_query_renders_as_a_table() {
        let mut s = shell();
        let out = s.handle("SELECT * FROM t ORDER BY a").unwrap();
        assert!(out.contains(" a "));
        assert!(out.contains(" x "));
        assert!(out.contains("2 rows"));
    }

    #[test]
    fn blank_lines_and_comments_produce_nothing() {
        let mut s = shell();
        assert!(s.handle("").unwrap().is_empty());
        assert!(s.handle("   ").unwrap().is_empty());
        assert!(s.handle("-- just a note").unwrap().is_empty());
    }

    #[test]
    fn a_comment_above_a_statement_does_not_swallow_it() {
        // The comment and the statement arrive as one buffer from a script.
        let mut s = Shell::new();
        let out = s
            .handle("-- make a table\nCREATE TABLE c (a INT);")
            .unwrap();
        assert!(out.contains("created table c"));
    }

    #[test]
    fn several_statements_on_one_line_all_run() {
        let mut s = Shell::new();
        let out = s
            .handle("CREATE TABLE u (a INT); INSERT INTO u VALUES (1); SELECT count(*) FROM u")
            .unwrap();
        assert!(out.contains("created table u"));
        assert!(out.contains("inserted 1 rows"));
        assert!(out.contains("1 row"));
    }

    #[test]
    fn listing_tables_shows_row_counts_and_where_they_live() {
        let mut s = shell();
        assert!(s.handle("\\dt").unwrap().contains("t "));
        assert!(s.handle("\\dt").unwrap().contains("memory"));
        assert!(Shell::new().handle("\\dt").unwrap().contains("no tables"));
    }

    #[test]
    fn describing_a_table_shows_its_columns_and_statistics() {
        let mut s = shell();
        let out = s.handle("\\d t").unwrap();
        assert!(out.contains("INT64"));
        assert!(out.contains("UTF8"));
        assert!(out.contains("min=1 max=2"));
        assert!(s.handle("\\d nope").is_err());
        assert!(s.handle("\\d").is_err());
    }

    #[test]
    fn importing_a_csv_creates_a_table() {
        let dir = TempDir::new("import");
        let path = dir.file("data.csv");
        std::fs::write(&path, "x,y\n1,a\n2,b\n").unwrap();
        let mut s = Shell::new();
        let out = s.handle(&format!("\\import loaded {path}")).unwrap();
        assert!(out.contains("imported 2 rows"));
        assert!(s
            .handle("SELECT count(*) FROM loaded")
            .unwrap()
            .contains("1 row"));
    }

    #[test]
    fn saving_and_attaching_move_a_table_through_the_columnar_format() {
        let dir = TempDir::new("save");
        let path = dir.file("t.qfc");
        let mut s = shell();
        assert!(s
            .handle(&format!("\\save t {path}"))
            .unwrap()
            .contains("wrote 2 rows"));
        assert!(s.handle("\\dt").unwrap().contains("file"));

        let mut fresh = Shell::new();
        let out = fresh.handle(&format!("\\attach t {path}")).unwrap();
        assert!(out.contains("attached t (2 rows)"));
        assert!(fresh
            .handle("SELECT b FROM t WHERE a = 2")
            .unwrap()
            .contains(" y "));
    }

    #[test]
    fn explain_shows_a_plan_both_ways() {
        let mut s = shell();
        let a = s.handle("EXPLAIN SELECT a FROM t WHERE a > 1").unwrap();
        let b = s.handle("\\explain SELECT a FROM t WHERE a > 1").unwrap();
        assert!(a.contains("Scan: t"));
        assert_eq!(a, b);
        assert!(s.handle("\\explain").is_err());
    }

    #[test]
    fn explain_analyze_reports_counters() {
        let mut s = shell();
        let out = s.handle("EXPLAIN ANALYZE SELECT count(*) FROM t").unwrap();
        assert!(out.contains("rows="));
        assert!(out.contains("rows returned"));
    }

    #[test]
    fn timing_toggles_and_annotates_queries() {
        let mut s = shell();
        assert!(!s.handle("SELECT 1").unwrap().contains("time:"));
        assert!(s.handle("\\timing").unwrap().contains("on"));
        assert!(s.handle("SELECT 1").unwrap().contains("time:"));
        assert!(s.handle("\\timing").unwrap().contains("off"));
        assert!(!s.handle("SELECT 1").unwrap().contains("time:"));
    }

    #[test]
    fn help_and_quit_are_recognised() {
        let mut s = Shell::new();
        for c in ["\\help", "\\h", "\\?"] {
            assert!(s.handle(c).unwrap().contains("queryforge"));
        }
        assert!(s.handle("\\q").unwrap().is_empty());
    }

    #[test]
    fn an_unknown_command_or_a_missing_argument_is_reported() {
        let mut s = Shell::new();
        assert!(s
            .handle("\\nope")
            .unwrap_err()
            .to_string()
            .contains("unknown command"));
        assert!(s
            .handle("\\import")
            .unwrap_err()
            .to_string()
            .contains("usage"));
        assert!(s
            .handle("\\save t")
            .unwrap_err()
            .to_string()
            .contains("usage"));
        assert!(s.handle("\\attach t").is_err());
    }

    #[test]
    fn a_failing_query_returns_an_error_rather_than_printing_one() {
        let mut s = shell();
        let err = s.handle("SELECT nope FROM t").unwrap_err();
        assert!(err.to_string().contains("no such column"));
    }

    #[test]
    fn the_session_is_reachable_for_scripted_setup() {
        let mut s = Shell::new();
        assert!(s.session().tables().is_empty());
    }
}
