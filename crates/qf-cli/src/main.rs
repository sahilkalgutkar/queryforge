//! `queryforge` — the command-line front end.
//!
//! ```text
//! queryforge                          start the shell
//! queryforge -c "SELECT 1"            run one statement and exit
//! queryforge script.sql               run a file
//! queryforge bench [--rows N]         build a dataset and measure the optimiser
//!
//!   --import <table>=<file.csv>       load a CSV before doing anything else
//!   --attach <table>=<file.qfc>       register a columnar file as a table
//! ```

mod bench;
mod render;
mod repl;
mod shell;

use qf_common::{Error, Result};
use shell::Shell;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<()> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", shell::HELP);
        return Ok(());
    }
    if args.first().is_some_and(|a| a == "bench") {
        return run_bench(&args[1..]);
    }

    let mut shell = Shell::new();
    let mut rest: Vec<&String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--import" | "--attach" => {
                let spec = args
                    .get(i + 1)
                    .ok_or_else(|| Error::plan(format!("{} needs <table>=<file>", args[i])))?;
                let (name, path) = spec
                    .split_once('=')
                    .ok_or_else(|| Error::plan(format!("expected <table>=<file>, got `{spec}`")))?;
                if args[i] == "--import" {
                    shell.session().import_csv(name, path)?;
                } else {
                    shell.session().attach(name, path)?;
                }
                i += 2;
            }
            _ => {
                rest.push(&args[i]);
                i += 1;
            }
        }
    }

    if let Some(pos) = rest.iter().position(|a| a.as_str() == "-c") {
        let sql = rest
            .get(pos + 1)
            .ok_or_else(|| Error::plan("-c needs a statement".to_string()))?;
        print!("{}", shell.handle(sql)?);
        return Ok(());
    }

    if let Some(path) = rest.first() {
        let text = std::fs::read_to_string(path.as_str())?;
        return run_script(&mut shell, &text);
    }

    repl::run(&mut shell)?;
    Ok(())
}

/// Runs a file, statement by statement, stopping at the first failure so a
/// broken script does not carry on against a half-built catalog.
fn run_script(shell: &mut Shell, text: &str) -> Result<()> {
    let mut buffer = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        // A standalone comment or blank line between statements contributes
        // nothing. Buffering it would make the next line look like a
        // continuation, which is how a `\`-command after a comment ends up
        // being parsed as SQL.
        if buffer.is_empty() && (trimmed.is_empty() || trimmed.starts_with("--")) {
            continue;
        }
        if buffer.is_empty() && trimmed.starts_with('\\') {
            print!("{}", shell.handle(trimmed)?);
            continue;
        }
        if !buffer.is_empty() {
            buffer.push('\n');
        }
        buffer.push_str(line);
        if buffer.trim_end().ends_with(';') {
            let statement = std::mem::take(&mut buffer);
            print!("{}", shell.handle(&statement)?);
        }
    }
    if !buffer.trim().is_empty() {
        print!("{}", shell.handle(&buffer)?);
    }
    Ok(())
}

fn run_bench(args: &[String]) -> Result<()> {
    let mut rows = bench::DEFAULT_ROWS;
    if let Some(pos) = args.iter().position(|a| a == "--rows") {
        rows = args
            .get(pos + 1)
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| Error::plan("--rows needs a number".to_string()))?;
    }
    let dir = std::env::temp_dir().join(format!("queryforge-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let result = (|| {
        eprintln!("building {rows} rows…");
        let mut session = bench::prepare(&dir, rows)?;
        eprintln!("running {} queries, optimised and not…", bench::CASES.len());
        let measurements = bench::run(&mut session)?;
        print!("{}", bench::report(rows, &measurements));
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(tag: &str) -> TempDir {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "queryforge-cli-{tag}-{}-{}",
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

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn help_is_printed_for_either_flag() {
        assert!(run(&args(&["--help"])).is_ok());
        assert!(run(&args(&["-h"])).is_ok());
    }

    #[test]
    fn a_single_statement_runs_with_dash_c() {
        assert!(run(&args(&["-c", "SELECT 1 + 1"])).is_ok());
    }

    #[test]
    fn a_bad_statement_comes_back_as_an_error() {
        assert!(run(&args(&["-c", "SELECT nope FROM missing"])).is_err());
        assert!(run(&args(&["-c"])).is_err());
    }

    #[test]
    fn a_csv_can_be_imported_before_the_query_runs() {
        let dir = TempDir::new("import");
        let csv = dir.file("d.csv");
        std::fs::write(&csv, "a,b\n1,x\n2,y\n").unwrap();
        assert!(run(&args(&[
            "--import",
            &format!("d={csv}"),
            "-c",
            "SELECT count(*) FROM d"
        ]))
        .is_ok());
    }

    #[test]
    fn a_malformed_preload_specification_is_reported() {
        assert!(run(&args(&["--import", "nocolon"]))
            .unwrap_err()
            .to_string()
            .contains("<table>=<file>"));
        assert!(run(&args(&["--import"])).is_err());
        assert!(run(&args(&["--attach", "t=/nonexistent.qfc"])).is_err());
    }

    #[test]
    fn a_qfc_file_can_be_attached_before_the_query_runs() {
        let dir = TempDir::new("attach");
        let csv = dir.file("d.csv");
        let qfc = dir.file("d.qfc");
        std::fs::write(&csv, "a\n1\n2\n").unwrap();
        let mut s = Shell::new();
        s.session().import_csv("d", &csv).unwrap();
        s.session().save_table("d", &qfc).unwrap();

        assert!(run(&args(&[
            "--attach",
            &format!("d={qfc}"),
            "-c",
            "SELECT count(*) FROM d"
        ]))
        .is_ok());
    }

    #[test]
    fn a_script_file_runs_statement_by_statement() {
        let dir = TempDir::new("script");
        let path = dir.file("s.sql");
        std::fs::write(
            &path,
            "-- set up\nCREATE TABLE t (a INT);\nINSERT INTO t\n  VALUES (1), (2);\n\\dt\nSELECT count(*) FROM t;\n",
        )
        .unwrap();
        assert!(run(&args(&[&path])).is_ok());
    }

    #[test]
    fn a_script_stops_at_its_first_failure() {
        let dir = TempDir::new("bad-script");
        let path = dir.file("s.sql");
        std::fs::write(&path, "SELECT nope FROM missing;\nSELECT 1;\n").unwrap();
        assert!(run(&args(&[&path])).is_err());
    }

    #[test]
    fn a_scripts_last_statement_runs_without_a_trailing_semicolon() {
        let mut shell = Shell::new();
        assert!(run_script(&mut shell, "CREATE TABLE t (a INT)").is_ok());
        assert!(shell.session().catalog().contains("t"));
    }

    #[test]
    fn a_missing_script_file_is_reported() {
        assert!(run(&args(&["/nonexistent/script.sql"])).is_err());
    }

    #[test]
    fn the_benchmark_runs_at_a_small_size() {
        assert!(run_bench(&args(&["--rows", "5000"])).is_ok());
        assert!(run_bench(&args(&["--rows", "not-a-number"])).is_err());
    }
}
