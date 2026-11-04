use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use crate::ast::Stmt;
use crate::debug::DebugCtl;
use crate::error::{MatError, MatResult};
use crate::plot::{Figure, FigureWindowStyle};
use crate::value::{Matrix, Value};

pub type Builtin = fn(&mut Session, &[Value]) -> MatResult<Value>;
pub type MatBin = fn(&Matrix, &Matrix) -> MatResult<Matrix>;

#[derive(Debug, Clone)]
pub struct UserFunc {
    pub params: Vec<String>,
    pub outputs: Vec<String>,
    pub body: Vec<Stmt>,
    /// The `.m` file the function was loaded from, if any.
    pub file: Option<PathBuf>,
}

pub struct Session {
    pub vars: HashMap<String, Value>,
    pub funcs: HashMap<String, UserFunc>,
    pub builtins: HashMap<String, Builtin>,
    /// Folders searched for `.m` files after the current folder, first entry first.
    pub search_path: Vec<PathBuf>,
    pub(crate) path_funcs: Vec<String>,
    pub(crate) path_scripts: HashMap<String, (PathBuf, Vec<Stmt>)>,
    pub(crate) path_errors: HashMap<String, String>,
    pub(crate) path_lookup: HashMap<String, Option<PathBuf>>,
    /// The file being executed, for `mfilename`.
    pub current_file: Option<PathBuf>,
    pub figures: Vec<Figure>,
    pub current_figure: usize,
    pub output: String,
    /// Set by `home`. The GUI replaces the console instead of appending.
    pub output_homed: bool,
    /// When set, `plot`, `stem`, `step`, and `bode` open a new figure if the
    /// current one already has curves and `hold` is off.
    pub new_figure_on_plot: bool,
    /// MATLAB root property `DefaultFigureWindowStyle`.
    pub default_figure_window_style: FigureWindowStyle,
    pub nargout: usize,
    /// Min and max real value seen for each assigned `fi` name.
    pub fi_seen: BTreeMap<String, (f64, f64)>,
    /// Samples collected by `NumericTypeScope`.
    pub numeric_type_scope: Vec<f64>,
    pub rng: u64,
    pub ldiv_hook: Option<MatBin>,
    pub rdiv_hook: Option<MatBin>,
    pub(crate) scopes: Vec<HashMap<String, Value>>,
    pub(crate) in_func: bool,
    index_rows: usize,
    index_cols: usize,
    index_dim: usize,
    index_count: usize,
    pub debug: Option<Arc<DebugCtl>>,
}

impl Session {
    pub fn new() -> Self {
        let mut session = Self {
            vars: HashMap::new(),
            funcs: HashMap::new(),
            builtins: HashMap::new(),
            search_path: Vec::new(),
            path_funcs: Vec::new(),
            path_scripts: HashMap::new(),
            path_errors: HashMap::new(),
            path_lookup: HashMap::new(),
            current_file: None,
            figures: vec![Figure::new()],
            current_figure: 0,
            output: String::new(),
            output_homed: false,
            new_figure_on_plot: false,
            default_figure_window_style: FigureWindowStyle::Docked,
            nargout: 1,
            fi_seen: BTreeMap::new(),
            numeric_type_scope: Vec::new(),
            rng: 0x1234_5678_9abc_def0,
            ldiv_hook: None,
            rdiv_hook: None,
            scopes: Vec::new(),
            in_func: false,
            index_rows: 0,
            index_cols: 0,
            index_dim: 0,
            index_count: 0,
            debug: None,
        };
        session.install_constants();
        session
    }

    fn install_constants(&mut self) {
        for name in [
            "pi", "i", "j", "inf", "Inf", "nan", "NaN", "true", "false",
        ] {
            self.install_constant(name);
        }
    }

    fn install_constant(&mut self, name: &str) {
        let value = match name {
            "pi" => Value::scalar(std::f64::consts::PI),
            "i" | "j" => Value::complex(0.0, 1.0),
            "inf" | "Inf" => Value::scalar(f64::INFINITY),
            "nan" | "NaN" => Value::scalar(f64::NAN),
            "true" => Value::logical(true),
            "false" => Value::logical(false),
            _ => return,
        };
        self.vars.insert(name.into(), value);
    }

    pub fn register_builtin(&mut self, name: &str, func: Builtin) {
        self.builtins.insert(name.to_string(), func);
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v);
            }
        }
        if !self.in_func {
            return self.vars.get(name);
        }
        None
    }

    pub fn insert(&mut self, name: String, value: Value) {
        if let Value::Fi(fi) = &value {
            let mut low = f64::INFINITY;
            let mut high = f64::NEG_INFINITY;
            for raw in &fi.raw {
                let real = fi.to_f64(*raw);
                low = low.min(real);
                high = high.max(real);
            }
            let entry = self.fi_seen.entry(name.clone()).or_insert((low, high));
            entry.0 = entry.0.min(low);
            entry.1 = entry.1.max(high);
        }
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, value);
        } else {
            self.vars.insert(name, value);
        }
    }

    /// Removes workspace variables and restores `pi`, `i`, `true`, and the other constants.
    pub fn styled_figure(&self) -> Figure {
        let mut figure = Figure::new();
        figure.window_style = self.default_figure_window_style;
        figure
    }

    pub fn clear_workspace(&mut self) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.clear();
            return;
        }
        self.vars.clear();
        self.fi_seen.clear();
        self.install_constants();
    }

    /// Removes one name. A cleared constant comes back, as it does in MATLAB.
    pub fn clear_name(&mut self, name: &str) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.remove(name);
            return;
        }
        self.vars.remove(name);
        self.funcs.remove(name);
        self.install_constant(name);
    }

    pub fn writeln(&mut self, text: &str) {
        self.output.push_str(text);
        if !text.ends_with('\n') {
            self.output.push('\n');
        }
    }

    pub fn write_raw(&mut self, text: &str) {
        self.output.push_str(text);
    }

    pub fn rand_f64(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        if x == 0 {
            x = 0x9e37_79b9_7f4a_7c15;
        }
        self.rng = x;
        ((x >> 11) as f64) / ((1u64 << 53) as f64)
    }

    pub fn current_figure_mut(&mut self) -> &mut Figure {
        if self.figures.is_empty() {
            self.figures.push(self.styled_figure());
            self.current_figure = 0;
        }
        if self.current_figure >= self.figures.len() {
            self.current_figure = self.figures.len() - 1;
        }
        &mut self.figures[self.current_figure]
    }

    pub(crate) fn lookup_func_env(&self) -> BTreeMap<String, Value> {
        let mut env = BTreeMap::new();
        if !self.in_func {
            for (k, v) in &self.vars {
                env.insert(k.clone(), v.clone());
            }
        }
        for scope in &self.scopes {
            for (k, v) in scope {
                env.insert(k.clone(), v.clone());
            }
        }
        env
    }

    pub(crate) fn with_index_context<T>(
        &mut self,
        rows: usize,
        cols: usize,
        count: usize,
        dim: usize,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let saved = (self.index_rows, self.index_cols, self.index_count, self.index_dim);
        self.index_rows = rows;
        self.index_cols = cols;
        self.index_count = count;
        self.index_dim = dim;
        let out = f(self);
        self.index_rows = saved.0;
        self.index_cols = saved.1;
        self.index_count = saved.2;
        self.index_dim = saved.3;
        out
    }

    pub(crate) fn end_index(&self) -> MatResult<f64> {
        if self.index_count == 0 {
            return Err(MatError::new("'end' is only valid inside an index"));
        }
        let n = if self.index_count == 1 {
            self.index_rows * self.index_cols
        } else if self.index_dim == 0 {
            self.index_rows
        } else {
            self.index_cols
        };
        Ok(n as f64)
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}
