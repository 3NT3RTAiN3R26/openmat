use std::path::PathBuf;

use crate::ast::{Stmt, StmtKind};
use crate::error::{MatError, MatResult};
use crate::parser::parse;
use crate::session::{Session, UserFunc};
use crate::value::{Matrix, Value};

fn is_file_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_alphabetic() || c == '_')
        && chars.all(|c| c.is_alphanumeric() || c == '_')
}

impl Session {
    /// Forgets functions and scripts read from the search path so edited files load again.
    pub fn unload_path_files(&mut self) {
        for name in std::mem::take(&mut self.path_funcs) {
            self.funcs.remove(&name);
        }
        self.path_scripts.clear();
        self.path_errors.clear();
    }

    fn find_on_path(&self, name: &str) -> Option<PathBuf> {
        let file = format!("{name}.m");
        std::env::current_dir()
            .ok()
            .into_iter()
            .chain(self.search_path.iter().cloned())
            .map(|dir| dir.join(&file))
            .find(|candidate| candidate.is_file())
    }

    /// True when `name` is a builtin, a defined function, or an `.m` file in the current folder or on the path.
    pub(crate) fn resolves(&mut self, name: &str) -> bool {
        if self.builtins.contains_key(name)
            || self.funcs.contains_key(name)
            || self.path_scripts.contains_key(name)
            || self.path_errors.contains_key(name)
        {
            return true;
        }
        if !is_file_name(name) {
            return false;
        }
        let Some(file) = self.find_on_path(name) else {
            return false;
        };
        let loaded = std::fs::read_to_string(&file)
            .map_err(|err| MatError::new(err.to_string()))
            .and_then(|src| parse(&src));
        match loaded {
            Ok(program) => self.install_path_file(name, program),
            Err(err) => {
                self.path_errors
                    .insert(name.to_string(), format!("{}: {err}", file.display()));
            }
        }
        true
    }

    fn install_path_file(&mut self, name: &str, program: Vec<Stmt>) {
        let is_function_file = matches!(
            program.first().map(|stmt| &stmt.kind),
            Some(StmtKind::Function { .. })
        );
        if !is_function_file {
            self.path_scripts.insert(name.to_string(), program);
            return;
        }
        for (index, stmt) in program.iter().enumerate() {
            if let StmtKind::Function {
                name: declared,
                outputs,
                params,
                body,
            } = &stmt.kind
            {
                // The file name wins over the declared name of the main function.
                let key = if index == 0 { name } else { declared };
                self.funcs.insert(
                    key.to_string(),
                    UserFunc {
                        params: params.clone(),
                        outputs: outputs.clone(),
                        body: body.clone(),
                    },
                );
                self.path_funcs.push(key.to_string());
            }
        }
    }

    /// Runs a script from the path, or reports why its file could not be loaded.
    pub(crate) fn run_path_entry(&mut self, name: &str) -> Option<MatResult<Value>> {
        if let Some(message) = self.path_errors.get(name) {
            return Some(Err(MatError::new(message.clone())));
        }
        let program = self.path_scripts.get(name)?.clone();
        Some(
            self.eval_program(&program)
                .map(|()| Value::Matrix(Matrix::empty())),
        )
    }
}
