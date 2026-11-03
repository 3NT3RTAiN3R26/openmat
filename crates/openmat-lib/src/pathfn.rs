use std::path::{Path, PathBuf};

use openmat_core::{MatError, MatResult, Session, Value};

use crate::util::{char_arg, empty};

const PATH_SEP: char = if cfg!(windows) { ';' } else { ':' };

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("addpath", addpath as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("rmpath", rmpath),
        ("path", path),
        ("genpath", genpath),
        ("pathsep", pathsep),
        ("filesep", filesep),
        ("pwd", pwd),
        ("cd", cd),
    ] {
        session.register_builtin(name, func);
    }
}

fn same_dir(a: &Path, b: &Path) -> bool {
    if cfg!(windows) {
        a.to_string_lossy().to_lowercase() == b.to_string_lossy().to_lowercase()
    } else {
        a == b
    }
}

fn joined(session: &Session) -> String {
    session
        .search_path
        .iter()
        .map(|dir| dir.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(&PATH_SEP.to_string())
}

/// The previous path for `old = addpath(...)`. Statement calls return nothing.
fn previous(session: &Session, old: String) -> Value {
    if session.nargout == 0 {
        empty()
    } else {
        Value::Char(old)
    }
}

fn absolute(dir: &str) -> PathBuf {
    let trimmed = dir.trim_end_matches(['/', '\\']);
    let dir = if trimmed.is_empty() { dir } else { trimmed };
    std::path::absolute(dir).unwrap_or_else(|_| PathBuf::from(dir))
}

fn split_list(text: &str) -> impl Iterator<Item = &str> {
    text.split(PATH_SEP).filter(|part| !part.is_empty())
}

fn addpath(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    if args.is_empty() {
        return Err(MatError::new("addpath expects at least one folder"));
    }
    let old = joined(session);
    let mut append = false;
    let mut folders: Vec<PathBuf> = Vec::new();
    for (index, arg) in args.iter().enumerate() {
        match arg {
            Value::Char(text) => match text.as_str() {
                "-begin" => append = false,
                "-end" => append = true,
                "-frozen" => {}
                _ => {
                    for part in split_list(text) {
                        let dir = absolute(part);
                        if dir.is_dir() {
                            if !folders.iter().any(|known| same_dir(known, &dir)) {
                                folders.push(dir);
                            }
                        } else {
                            session.writeln(&format!("Warning: Name is nonexistent or not a folder: {part}"));
                        }
                    }
                }
            },
            other if index > 0 => match other.real_scalar()? {
                v if v == 0.0 => append = false,
                v if v == 1.0 => append = true,
                _ => return Err(MatError::new("addpath position must be 0, 1, '-begin' or '-end'")),
            },
            other => {
                return Err(MatError::new(format!(
                    "addpath expects a folder name, got {}",
                    other.type_name()
                )))
            }
        }
    }
    session
        .search_path
        .retain(|known| !folders.iter().any(|dir| same_dir(known, dir)));
    if append {
        session.search_path.extend(folders);
    } else {
        folders.append(&mut session.search_path);
        session.search_path = folders;
    }
    session.unload_path_files();
    Ok(previous(session, old))
}

fn rmpath(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    if args.is_empty() {
        return Err(MatError::new("rmpath expects at least one folder"));
    }
    let old = joined(session);
    for index in 0..args.len() {
        let text = char_arg(args, index)?;
        for part in split_list(&text) {
            let dir = absolute(part);
            let before = session.search_path.len();
            session.search_path.retain(|known| !same_dir(known, &dir));
            if session.search_path.len() == before {
                session.writeln(&format!("Warning: \"{part}\" not found in path."));
            }
        }
    }
    session.unload_path_files();
    Ok(previous(session, old))
}

fn path(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    if args.is_empty() {
        return Ok(Value::Char(joined(session)));
    }
    let old = joined(session);
    let mut list: Vec<PathBuf> = Vec::new();
    for index in 0..args.len() {
        let text = char_arg(args, index)?;
        for part in split_list(&text) {
            let dir = absolute(part);
            if !list.iter().any(|known| same_dir(known, &dir)) {
                list.push(dir);
            }
        }
    }
    session.search_path = list;
    session.unload_path_files();
    Ok(previous(session, old))
}

fn genpath(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let root = absolute(&char_arg(args, 0)?);
    let mut out = Vec::new();
    collect_dirs(&root, &mut out)?;
    Ok(Value::Char(
        out.iter()
            .map(|dir| dir.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(&PATH_SEP.to_string()),
    ))
}

fn collect_dirs(dir: &Path, out: &mut Vec<PathBuf>) -> MatResult<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    out.push(dir.to_path_buf());
    let entries = std::fs::read_dir(dir).map_err(|err| MatError::new(err.to_string()))?;
    let mut children: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|child| child.is_dir())
        .filter(|child| {
            let name = child.file_name().and_then(|n| n.to_str()).unwrap_or("");
            !(name.starts_with(['.', '@', '+']) || name == "private" || name == "resources")
        })
        .collect();
    children.sort();
    for child in children {
        collect_dirs(&child, out)?;
    }
    Ok(())
}

fn pathsep(_: &mut Session, _: &[Value]) -> MatResult<Value> {
    Ok(Value::Char(PATH_SEP.to_string()))
}

fn filesep(_: &mut Session, _: &[Value]) -> MatResult<Value> {
    Ok(Value::Char(std::path::MAIN_SEPARATOR.to_string()))
}

fn current_dir() -> MatResult<Value> {
    let dir = std::env::current_dir().map_err(|err| MatError::new(err.to_string()))?;
    Ok(Value::Char(dir.to_string_lossy().into_owned()))
}

fn pwd(_: &mut Session, _: &[Value]) -> MatResult<Value> {
    current_dir()
}

fn cd(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    if args.is_empty() {
        return current_dir();
    }
    let target = char_arg(args, 0)?;
    std::env::set_current_dir(&target)
        .map_err(|_| MatError::new(format!("Cannot CD to {target} (Name is nonexistent or not a directory).")))?;
    session.unload_path_files();
    if session.nargout == 0 {
        Ok(empty())
    } else {
        current_dir()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use crate::exec;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("openmat_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn lit(dir: &std::path::Path) -> String {
        dir.to_string_lossy().replace('\'', "''")
    }

    #[test]
    fn function_file_is_found_after_addpath_and_lost_after_rmpath() {
        let dir = temp_dir("fn");
        fs::write(dir.join("triple.m"), "function y = triple(x)\ny = 3 * x;\nend\n").unwrap();
        let d = lit(&dir);
        let session = exec(&format!("addpath('{d}'); y = triple(4);")).unwrap();
        assert_eq!(session.get("y").unwrap().real_scalar().unwrap(), 12.0);
        let err = exec(&format!("addpath('{d}'); rmpath('{d}'); triple(1);")).err().expect("error");
        assert!(err.to_string().contains("undefined"), "{err}");
        assert!(exec("triple(1);").is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn file_name_wins_over_declared_function_name() {
        let dir = temp_dir("name");
        fs::write(dir.join("outer.m"), "function r = inner(x)\nr = x + 1;\nend\n").unwrap();
        let session = exec(&format!("addpath('{}'); r = outer(1);", lit(&dir))).unwrap();
        assert_eq!(session.get("r").unwrap().real_scalar().unwrap(), 2.0);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn script_on_path_runs_in_the_calling_workspace() {
        let dir = temp_dir("script");
        fs::write(dir.join("setup_vars.m"), "a = 5;\nb = a * 2;\n").unwrap();
        let session = exec(&format!("addpath('{}'); setup_vars; c = b + 1;", lit(&dir))).unwrap();
        assert_eq!(session.get("c").unwrap().real_scalar().unwrap(), 11.0);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn syntax_error_in_a_path_file_is_reported() {
        let dir = temp_dir("broken");
        fs::write(dir.join("broken_fn.m"), "x = (1 + ;\n").unwrap();
        let err = exec(&format!("addpath('{}'); broken_fn;", lit(&dir))).err().expect("error");
        assert!(err.to_string().contains("broken_fn.m"), "{err}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn begin_end_order_duplicates_and_old_path() {
        let a = temp_dir("order_a");
        let b = temp_dir("order_b");
        let c = temp_dir("order_c");
        let src = format!(
            "addpath('{a}'); old = addpath('{b}'); addpath('{c}', '-end'); addpath('{b}', '-end'); p = path;",
            a = lit(&a),
            b = lit(&b),
            c = lit(&c)
        );
        let session = exec(&src).unwrap();
        let dirs: Vec<PathBuf> = session.search_path.clone();
        assert_eq!(dirs, vec![a.clone(), c.clone(), b.clone()]);
        match session.get("old").unwrap() {
            openmat_core::Value::Char(text) => assert_eq!(text, &a.to_string_lossy()),
            other => panic!("{other}"),
        }
        let both = format!("addpath('{}', '{}');", lit(&a), lit(&b));
        assert_eq!(exec(&both).unwrap().search_path, vec![a.clone(), b.clone()]);
        for dir in [a, b, c] {
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn missing_folder_warns_and_is_skipped() {
        let missing = std::env::temp_dir().join("openmat_does_not_exist_xyz");
        let session = exec(&format!("addpath('{}');", lit(&missing))).unwrap();
        assert!(session.search_path.is_empty());
        assert!(session.output.contains("Warning"), "{}", session.output);
    }

    #[test]
    fn earlier_path_entry_shadows_later_one_and_edits_reload() {
        let first = temp_dir("shadow_first");
        let second = temp_dir("shadow_second");
        fs::write(first.join("which_one.m"), "function r = which_one()\nr = 1;\nend\n").unwrap();
        fs::write(second.join("which_one.m"), "function r = which_one()\nr = 2;\nend\n").unwrap();
        let (f, s) = (lit(&first), lit(&second));
        let session = exec(&format!("addpath('{s}'); addpath('{f}'); r = which_one();")).unwrap();
        assert_eq!(session.get("r").unwrap().real_scalar().unwrap(), 1.0);

        let mut session = exec(&format!("addpath('{f}'); r = which_one();")).unwrap();
        fs::write(first.join("which_one.m"), "function r = which_one()\nr = 7;\nend\n").unwrap();
        session.eval_source("r = which_one();").unwrap();
        assert_eq!(session.get("r").unwrap().real_scalar().unwrap(), 7.0);
        for dir in [first, second] {
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn genpath_lists_subfolders_but_skips_private_and_hidden() {
        let root = temp_dir("gen");
        for sub in ["lib", "lib/deep", "private", "@cls", "+pkg", ".git"] {
            fs::create_dir_all(root.join(sub)).unwrap();
        }
        let session = exec(&format!("p = genpath('{}');", lit(&root))).unwrap();
        let text = match session.get("p").unwrap() {
            openmat_core::Value::Char(text) => text.clone(),
            other => panic!("{other}"),
        };
        let parts: Vec<&str> = text.split(if cfg!(windows) { ';' } else { ':' }).collect();
        assert_eq!(parts.len(), 3, "{parts:?}");
        assert!(parts[1].ends_with("lib") && parts[2].ends_with("deep"));
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn bare_function_names_are_called_not_turned_into_handles() {
        let session = exec("s = pathsep; f = filesep;").unwrap();
        assert!(matches!(session.get("s"), Some(openmat_core::Value::Char(_))));
        assert!(matches!(session.get("f"), Some(openmat_core::Value::Char(_))));
    }

    #[test]
    fn path_set_and_statement_calls_print_nothing() {
        let a = temp_dir("pset_a");
        let session = exec(&format!("addpath('{}');", lit(&a))).unwrap();
        assert!(!session.output.contains("ans"), "{}", session.output);
        let session = exec(&format!("path('{}'); q = path;", lit(&a))).unwrap();
        assert_eq!(session.search_path, vec![a.clone()]);
        fs::remove_dir_all(&a).unwrap();
    }
}
