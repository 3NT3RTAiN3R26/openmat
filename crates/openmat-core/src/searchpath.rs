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

/// Functions from files are stored under their file path. A local function of that file is `path#name`.
fn file_key(file: &Path) -> String {
    file.to_string_lossy().into_owned()
}

/// `a.b.fn` is the file `+a/+b/fn.m`. A plain name is `name.m`. None when a segment is not a valid name.
fn relative_file(name: &str) -> Option<PathBuf> {
    let parts: Vec<&str> = name.split('.').collect();
    if !parts.iter().all(|part| is_file_name(part)) {
        return None;
    }
    let (last, packages) = parts.split_last()?;
    let mut file = PathBuf::new();
    for package in packages {
        file.push(format!("+{package}"));
    }
    file.push(format!("{last}.m"));
    Some(file)
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
    /// Forgets functions and scripts read from files so edited files load again.
    pub fn unload_path_files(&mut self) {
        for key in std::mem::take(&mut self.path_funcs) {
            self.funcs.remove(&key);
        }
        self.path_scripts.clear();
        self.path_errors.clear();
        self.path_lookup.clear();
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

    /// The `.m` file `name` is in the current folder or on the path. `pkg.fn` looks for `+pkg/fn.m`.
    pub fn which_file(&self, name: &str) -> Option<PathBuf> {
        let relative = relative_file(name)?;
        std::env::current_dir()
            .ok()
            .into_iter()
            .chain(self.search_path.iter().cloned())
            .map(|dir| dir.join(&relative))
            .find(|candidate| candidate.is_file())
    }

    /// The `private` folder whose functions the running file may call.
    fn private_dir(&self) -> Option<PathBuf> {
        let dir = self.current_file.as_ref()?.parent()?;
        if dir.file_name().is_some_and(|name| name == "private") {
            Some(dir.to_path_buf())
        } else {
            Some(dir.join("private"))
        }
    }

    fn private_file(&mut self, name: &str) -> Option<PathBuf> {
        if !is_file_name(name) {
            return None;
        }
        let dir = self.private_dir()?;
        let cache = format!("private\0{}\0{name}", dir.display());
        if let Some(hit) = self.path_lookup.get(&cache) {
            return hit.clone();
        }
        let file = dir.join(format!("{name}.m"));
        let found = file.is_file().then_some(file);
        self.path_lookup.insert(cache, found.clone());
        found
    }

    fn public_file(&mut self, name: &str) -> Option<PathBuf> {
        let cache = format!("path\0{name}");
        if let Some(hit) = self.path_lookup.get(&cache) {
            return hit.clone();
        }
        let found = self.which_file(name);
        self.path_lookup.insert(cache, found.clone());
        found
    }

    /// Finds what `name` calls and returns the key to run it by: a function defined in the
    /// script, a function local to the running file, a `private` function, a builtin, then
    /// a file in the current folder, on the path, or in a `+package` folder.
    pub(crate) fn resolve(&mut self, name: &str) -> Option<String> {
        if let Some(file) = &self.current_file {
            let local = format!("{}#{name}", file_key(file));
            if self.funcs.contains_key(&local) {
                return Some(local);
            }
        }
        if self.funcs.contains_key(name) {
            return Some(name.to_string());
        }
        if let Some(file) = self.private_file(name) {
            return Some(self.ensure_loaded(&file));
        }
        if self.builtins.contains_key(name) {
            return Some(name.to_string());
        }
        let file = self.public_file(name)?;
        Some(self.ensure_loaded(&file))
    }

    fn ensure_loaded(&mut self, file: &Path) -> String {
        let key = file_key(file);
        if self.funcs.contains_key(&key)
            || self.path_scripts.contains_key(&key)
            || self.path_errors.contains_key(&key)
        {
            return key;
        }
        let loaded = std::fs::read_to_string(file)
            .map_err(|err| MatError::new(err.to_string()))
            .and_then(|src| parse(&src));
        match loaded {
            Ok(program) => self.install_file(&key, file, program),
            Err(err) => {
                self.path_errors
                    .insert(key.clone(), format!("{}: {err}", file.display()));
            }
        }
        key
    }

    fn install_file(&mut self, key: &str, file: &Path, program: Vec<Stmt>) {
        let is_function_file = matches!(
            program.first().map(|stmt| &stmt.kind),
            Some(StmtKind::Function { .. })
        );
        if !is_function_file {
            self.path_scripts
                .insert(key.to_string(), (file.to_path_buf(), program));
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
                let entry = if index == 0 {
                    key.to_string()
                } else {
                    format!("{key}#{declared}")
                };
                self.funcs.insert(
                    entry.clone(),
                    UserFunc {
                        params: params.clone(),
                        outputs: outputs.clone(),
                        body: body.clone(),
                        file: Some(file.to_path_buf()),
                    },
                );
                self.path_funcs.push(entry);
            }
        }
    }

    /// Runs a script by its key, or reports why its file could not be loaded.
    pub(crate) fn run_path_entry(&mut self, key: &str) -> Option<MatResult<Value>> {
        if let Some(message) = self.path_errors.get(key) {
            return Some(Err(MatError::new(message.clone())));
        }
        let (file, program) = self.path_scripts.get(key)?.clone();
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
