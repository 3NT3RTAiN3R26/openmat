use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::process::ExitCode;

use clap::Parser;
use openmat_lib::Session;
use rustyline::DefaultEditor;

#[derive(Parser)]
#[command(
    name = "openmat",
    version,
    about = "Run MATLAB-compatible .m files. MATLAB is a trademark of MathWorks; openmat is not affiliated with MathWorks."
)]
struct Cli {
    /// Script to execute
    file: Option<std::path::PathBuf>,
    /// Evaluate one statement
    #[arg(short = 'e', long = "eval")]
    eval: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let mut session = openmat_lib::session();
    if let Some(code) = cli.eval {
        return finish(&mut session, &code);
    }
    if let Some(path) = cli.file {
        let code = match fs::read_to_string(&path) {
            Ok(code) => code,
            Err(err) => {
                eprintln!("cannot read {}: {err}", path.display());
                return ExitCode::from(1);
            }
        };
        return finish(&mut session, &code);
    }
    if io::stdin().is_terminal() {
        repl(&mut session);
        ExitCode::SUCCESS
    } else {
        let mut code = String::new();
        if io::stdin().read_to_string(&mut code).is_err() {
            return ExitCode::from(1);
        }
        finish(&mut session, &code)
    }
}

fn finish(session: &mut Session, code: &str) -> ExitCode {
    match session.eval_source(code) {
        Ok(()) => {
            print!("{}", session.output);
            let _ = io::stdout().flush();
            ExitCode::SUCCESS
        }
        Err(err) => {
            print!("{}", session.output);
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn repl(session: &mut Session) {
    let mut editor = match DefaultEditor::new() {
        Ok(editor) => editor,
        Err(err) => {
            eprintln!("repl is unavailable: {err}");
            return;
        }
    };
    println!("openmat 0.1.0. MATLAB is a trademark of MathWorks. openmat is not affiliated with MathWorks.");
    loop {
        match editor.readline(">> ") {
            Ok(line) => {
                if line.trim().is_empty() {
                    continue;
                }
                let _ = editor.add_history_entry(line.as_str());
                session.output.clear();
                match session.eval_source(&line) {
                    Ok(()) => print!("{}", session.output),
                    Err(err) => eprintln!("{err}"),
                }
            }
            Err(_) => break,
        }
    }
}
