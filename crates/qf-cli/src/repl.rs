//! The terminal loop: read a line, hand it to the shell, print what comes
//! back. Everything worth testing lives in `shell.rs`, which the tests drive
//! directly, so this file is excluded from coverage.

use crate::shell::Shell;
use std::io::{BufRead, Write};

/// Reads statements until end of input.
///
/// A statement can span lines: input accumulates until it ends with a
/// semicolon, and a backslash command runs immediately.
pub fn run(shell: &mut Shell) -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut buffer = String::new();

    writeln!(stdout, "queryforge — \\help for commands, \\q to quit")?;
    prompt(&mut stdout, false)?;

    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();

        if buffer.is_empty() && (trimmed == "\\q" || trimmed == "\\quit") {
            return Ok(());
        }
        // Same as in a script: a comment between statements is not the start
        // of one.
        if buffer.is_empty() && trimmed.starts_with("--") {
            prompt(&mut stdout, false)?;
            continue;
        }
        if buffer.is_empty() && trimmed.starts_with('\\') {
            emit(&mut stdout, shell.handle(trimmed));
            prompt(&mut stdout, false)?;
            continue;
        }

        if !buffer.is_empty() {
            buffer.push('\n');
        }
        buffer.push_str(&line);

        // A statement is complete at a semicolon. Without one, keep reading —
        // which is what makes a multi-line query possible.
        if !buffer.trim_end().ends_with(';') {
            if buffer.trim().is_empty() {
                buffer.clear();
                prompt(&mut stdout, false)?;
            } else {
                prompt(&mut stdout, true)?;
            }
            continue;
        }

        let statement = std::mem::take(&mut buffer);
        emit(&mut stdout, shell.handle(&statement));
        prompt(&mut stdout, false)?;
    }

    // Whatever was typed without a closing semicolon still runs at EOF, so a
    // piped script does not silently drop its last statement.
    if !buffer.trim().is_empty() {
        emit(&mut stdout, shell.handle(&buffer));
    }
    writeln!(stdout)?;
    Ok(())
}

fn emit(out: &mut impl Write, result: qf_common::Result<String>) {
    match result {
        Ok(text) => {
            if !text.is_empty() {
                let _ = write!(out, "{text}");
            }
        }
        Err(e) => {
            let _ = writeln!(out, "{e}");
        }
    }
}

fn prompt(out: &mut impl Write, continuation: bool) -> std::io::Result<()> {
    write!(out, "{}", if continuation { "  ...> " } else { "qf> " })?;
    out.flush()
}
