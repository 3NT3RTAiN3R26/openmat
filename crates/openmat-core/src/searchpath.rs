use std::path::{Path, PathBuf};

use crate::ast::{Stmt, StmtKind};
use crate::error::{MatError, MatResult};
use crate::parser::parse;
use crate::session::{Session, UserFunc};
use crate::value::{Matrix, Value};

const PATH_SEP: char = if cfg!(windows) { ';' } else { ':' };

fn is_file_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_alphabetic() || c == '_')
        && chars.all(|c| c.is_alphanumeric() || c == '_')
}

/// Where `savepath` writes the search path. `OPENMAT_PATHDEF` overrides the default location.
pub fn pathdef_file() -> Option<PathBuf> {
    if let Some(file) = std::env::var_os("OPENMAT_PATHDEF") {
        return Some(PathBuf::from(file));
    }
    let base = if cfg!(windows) {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
    }?;
    Some(base.join("openmat").join("pathdef.txt"))
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

    pub fn search_path_text(&self) -> String {
        self.search_path
            .iter()
            .map(|dir| dir.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(&PATH_SEP.to_string())
    }

    /// Writes the search path, one folder per line.
    pub fn save_path_to(&self, file: &Path) -> MatResult<()> {
        if let Some(parent) = file.parent() {
            std::fs::create_dir_all(parent).map_err(|err| MatError::new(err.to_string()))?;
        }
        let text: String = self
            .search_path
            .iter()
            .map(|dir| format!("{}\n", dir.display()))
            .collect();
        std::fs::write(file, text)
            .map_err(|err| MatError::new(format!("Unable to save path to {}: {err}", file.display())))
    }

    /// Replaces the search path with the saved one. Folders that no longer exist are dropped.
    pub fn load_path_from(&mut self, file: &Path) -> MatResult<()> {
        let text = match std::fs::read_to_string(file) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(err) => return Err(MatError::new(err.to_string())),
        };
        self.search_path = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(PathBuf::from)
            .filter(|dir| dir.is_dir())
            .collect();
        self.unload_path_files();
        Ok(())
    }

    /// Loads the path written by an earlier `savepath` or the GUI. A missing file is not an error.
    pub fn load_saved_path(&mut self) -> MatResult<()> {
        match pathdef_file() {
            Some(file) => self.load_path_from(&file),
            None => Ok(()),
        }
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

    /// The `.m` file `name` resolves to in the current folder or on the path.
    pub fn which_file(&self, name: &str) -> Option<PathBuf> {
        if !is_file_name(name) {
            return None;
        }
        self.find_on_path(name)
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
        let Some(file) = self.which_file(name) else {
            return false;
        };
        let loaded = std::fs::read_to_string(&file)
            .map_err(|err| MatError::new(err.to_string()))
            .and_then(|src| parse(&src));
        match loaded {
            Ok(program) => self.install_path_file(name, &file, program),
            Err(err) => {
                self.path_errors
                    .insert(name.to_string(), format!("{}: {err}", file.display()));
            }
        }
        true
    }

    fn install_path_file(&mut self, name: &str, file: &Path, program: Vec<Stmt>) {
        let is_function_file = matches!(
            program.first().map(|stmt| &stmt.kind),
            Some(StmtKind::Function { .. })
        );
        if !is_function_file {
            self.path_scripts
                .insert(name.to_string(), (file.to_path_buf(), program));
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
                        file: Some(file.to_path_buf()),
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
        let (file, program) = self.path_scripts.get(name)?.clone();
        Some(self.run_program_from(&file, &program))
    }

    fn run_program_from(&mut self, file: &Path, program: &[Stmt]) -> MatResult<Value> {
        let saved = self.current_file.replace(file.to_path_buf());
        let result = self.eval_program(program);
        self.current_file = saved;
        result.map(|()| Value::Matrix(Matrix::empty()))
    }

    /// Runs an `.m` file in the current workspace, as `run` does.
    pub fn run_file(&mut self, file: &Path) -> MatResult<()> {
        let file = if file.extension().is_none() {
            file.with_extension("m")
        } else {
            file.to_path_buf()
        };
        let src = std::fs::read_to_string(&file)
            .map_err(|_| MatError::new(format!("Cannot find file {}", file.display())))?;
        let program = parse(&src).map_err(|err| MatError::new(format!("{}: {err}", file.display())))?;
        self.run_program_from(&file, &program).map(|_| ())
    }
}
