use std::collections::BTreeMap;
use std::fmt;

use num_complex::Complex64;

use crate::ast::Expr;
use crate::error::{MatError, MatResult};

const EPS: f64 = 1e-9;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Complex64>,
    pub logical: bool,
}

impl Matrix {
    pub fn empty() -> Self {
        Self {
            rows: 0,
            cols: 0,
            data: Vec::new(),
            logical: false,
        }
    }

    pub fn scalar(v: f64) -> Self {
        Self::from_reals(1, 1, vec![v])
    }

    pub fn logical_scalar(v: bool) -> Self {
        let mut m = Self::scalar(if v { 1.0 } else { 0.0 });
        m.logical = true;
        m
    }

    pub fn from_reals(rows: usize, cols: usize, reals: Vec<f64>) -> Self {
        assert_eq!(reals.len(), rows * cols);
        Self {
            rows,
            cols,
            data: reals.into_iter().map(Complex64::from).collect(),
            logical: false,
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Complex64::new(0.0, 0.0); rows * cols],
            logical: false,
        }
    }

    pub fn index(rows: usize, r: usize, c: usize) -> usize {
        c * rows + r
    }

    pub fn get(&self, r: usize, c: usize) -> Complex64 {
        self.data[Self::index(self.rows, r, c)]
    }

    pub fn set(&mut self, r: usize, c: usize, v: Complex64) {
        let i = Self::index(self.rows, r, c);
        self.data[i] = v;
    }

    pub fn is_scalar(&self) -> bool {
        self.rows == 1 && self.cols == 1
    }

    pub fn is_real(&self) -> bool {
        self.data.iter().all(|z| z.im.abs() < 1e-8)
    }

    pub fn real_scalar(&self) -> MatResult<f64> {
        if !self.is_scalar() {
            return Err(MatError::new("expected a scalar"));
        }
        if !self.is_real() {
            return Err(MatError::new("expected a real scalar"));
        }
        Ok(self.data[0].re)
    }

    pub fn numel(&self) -> usize {
        self.rows * self.cols
    }

    pub fn column(&self, c: usize) -> Matrix {
        let mut data = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            data.push(self.get(r, c));
        }
        Matrix {
            rows: self.rows,
            cols: 1,
            data,
            logical: self.logical,
        }
    }

    pub fn transpose(&self, conjugate: bool) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        out.logical = self.logical;
        for r in 0..self.rows {
            for c in 0..self.cols {
                let mut v = self.get(r, c);
                if conjugate {
                    v = v.conj();
                }
                out.set(c, r, v);
            }
        }
        out
    }

    pub fn horzcat(&self, other: &Matrix) -> MatResult<Matrix> {
        if self.numel() == 0 {
            return Ok(other.clone());
        }
        if other.numel() == 0 {
            return Ok(self.clone());
        }
        if self.rows != other.rows {
            return Err(MatError::new(format!(
                "horizontal concatenation needs the same number of rows ({} vs {})",
                self.rows, other.rows
            )));
        }
        let mut out = Matrix::zeros(self.rows, self.cols + other.cols);
        for c in 0..self.cols {
            for r in 0..self.rows {
                out.set(r, c, self.get(r, c));
            }
        }
        for c in 0..other.cols {
            for r in 0..other.rows {
                out.set(r, self.cols + c, other.get(r, c));
            }
        }
        Ok(out)
    }

    pub fn vertcat(&self, other: &Matrix) -> MatResult<Matrix> {
        if self.numel() == 0 {
            return Ok(other.clone());
        }
        if other.numel() == 0 {
            return Ok(self.clone());
        }
        if self.cols != other.cols {
            return Err(MatError::new(format!(
                "vertical concatenation needs the same number of columns ({} vs {})",
                self.cols, other.cols
            )));
        }
        let mut out = Matrix::zeros(self.rows + other.rows, self.cols);
        for c in 0..self.cols {
            for r in 0..self.rows {
                out.set(r, c, self.get(r, c));
            }
            for r in 0..other.rows {
                out.set(self.rows + r, c, other.get(r, c));
            }
        }
        Ok(out)
    }

    pub fn map<F>(&self, mut f: F) -> Matrix
    where
        F: FnMut(Complex64) -> Complex64,
    {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().copied().map(&mut f).collect(),
            logical: false,
        }
    }

    pub fn ew(&self, other: &Matrix, f: impl Fn(Complex64, Complex64) -> Complex64) -> MatResult<Matrix> {
        if self.is_scalar() {
            return Ok(other.map(|b| f(self.data[0], b)));
        }
        if other.is_scalar() {
            return Ok(self.map(|a| f(a, other.data[0])));
        }
        let rows = broadcast_len(self.rows, other.rows).map_err(|_| {
            MatError::new(format!(
                "matrix dimensions must agree ({}x{} vs {}x{})",
                self.rows, self.cols, other.rows, other.cols
            ))
        })?;
        let cols = broadcast_len(self.cols, other.cols).map_err(|_| {
            MatError::new(format!(
                "matrix dimensions must agree ({}x{} vs {}x{})",
                self.rows, self.cols, other.rows, other.cols
            ))
        })?;
        let mut data = Vec::with_capacity(rows * cols);
        for c in 0..cols {
            for r in 0..rows {
                data.push(f(self.expanded(r, c), other.expanded(r, c)));
            }
        }
        Ok(Matrix {
            rows,
            cols,
            data,
            logical: false,
        })
    }

    fn expanded(&self, row: usize, col: usize) -> Complex64 {
        let row = if self.rows == 1 { 0 } else { row };
        let col = if self.cols == 1 { 0 } else { col };
        self.get(row, col)
    }

    pub fn matmul(&self, other: &Matrix) -> MatResult<Matrix> {
        if self.is_scalar() || other.is_scalar() {
            return self.ew(other, |a, b| a * b);
        }
        if self.cols != other.rows {
            return Err(MatError::new(format!(
                "inner matrix dimensions must agree ({}x{} * {}x{})",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut out = Matrix::zeros(self.rows, other.cols);
        for c in 0..other.cols {
            for r in 0..self.rows {
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..self.cols {
                    acc += self.get(r, k) * other.get(k, c);
                }
                out.set(r, c, acc);
            }
        }
        Ok(out)
    }

    pub fn reals(&self) -> MatResult<Vec<f64>> {
        if !self.is_real() {
            return Err(MatError::new("expected a real matrix"));
        }
        Ok(self.data.iter().map(|z| z.re).collect())
    }
}

#[derive(Debug, Clone)]
pub struct Fi {
    pub signed: bool,
    pub word: u32,
    pub frac: i32,
    pub rows: usize,
    pub cols: usize,
    pub raw: Vec<i64>,
}

impl Fi {
    pub fn quantize(val: f64, signed: bool, word: u32, frac: i32) -> MatResult<i64> {
        if !(1..=62).contains(&word) {
            return Err(MatError::new("word length must be between 1 and 62"));
        }
        let scale = 2f64.powi(frac);
        let mut raw = (val * scale).round();
        if !raw.is_finite() {
            return Err(MatError::new("cannot quantize a non-finite value"));
        }
        let (min, max) = if signed {
            (-(1i64 << (word - 1)), (1i64 << (word - 1)) - 1)
        } else {
            (0, (1i64 << word) - 1)
        };
        raw = raw.clamp(min as f64, max as f64);
        Ok(raw as i64)
    }

    pub fn from_matrix(m: &Matrix, signed: bool, word: u32, frac: i32) -> MatResult<Self> {
        let reals = m.reals()?;
        let raw = reals
            .into_iter()
            .map(|v| Self::quantize(v, signed, word, frac))
            .collect::<MatResult<Vec<_>>>()?;
        Ok(Self {
            signed,
            word,
            frac,
            rows: m.rows,
            cols: m.cols,
            raw,
        })
    }

    pub fn to_f64(&self, raw: i64) -> f64 {
        raw as f64 / 2f64.powi(self.frac)
    }

    pub fn to_matrix(&self) -> Matrix {
        let data = self
            .raw
            .iter()
            .map(|r| Complex64::from(self.to_f64(*r)))
            .collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
            logical: false,
        }
    }

    pub fn hex(&self) -> Vec<String> {
        let digits = ((self.word as usize) + 3) / 4;
        let mask = if self.word >= 64 {
            u64::MAX
        } else {
            (1u64 << self.word) - 1
        };
        self.raw
            .iter()
            .map(|r| format!("{:0width$x}", (*r as u64) & mask, width = digits))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub enum System {
    Tf { num: Vec<f64>, den: Vec<f64> },
    Ss {
        a: Matrix,
        b: Matrix,
        c: Matrix,
        d: Matrix,
    },
}

#[derive(Debug, Clone)]
pub enum FuncHandle {
    Anon {
        params: Vec<String>,
        body: Expr,
        env: BTreeMap<String, Value>,
    },
    Named(String),
}

#[derive(Debug, Clone)]
pub enum Value {
    Matrix(Matrix),
    Char(String),
    Cell {
        rows: usize,
        cols: usize,
        data: Vec<Value>,
    },
    Struct(BTreeMap<String, Value>),
    Func(FuncHandle),
    Fi(Fi),
    System(System),
    Multi(Vec<Value>),
}

impl Value {
    pub fn scalar(v: f64) -> Self {
        Value::Matrix(Matrix::scalar(v))
    }

    pub fn logical(v: bool) -> Self {
        Value::Matrix(Matrix::logical_scalar(v))
    }

    pub fn complex(re: f64, im: f64) -> Self {
        Value::Matrix(Matrix {
            rows: 1,
            cols: 1,
            data: vec![Complex64::new(re, im)],
            logical: false,
        })
    }

    pub fn as_matrix(&self) -> MatResult<&Matrix> {
        match self {
            Value::Matrix(m) => Ok(m),
            Value::Fi(_) => Err(MatError::new(
                "fixed-point value needs double() before this operation",
            )),
            other => Err(MatError::new(format!(
                "expected a numeric matrix, got {}",
                other.type_name()
            ))),
        }
    }

    pub fn into_matrix(self) -> MatResult<Matrix> {
        match self {
            Value::Matrix(m) => Ok(m),
            Value::Fi(fi) => Ok(fi.to_matrix()),
            other => Err(MatError::new(format!(
                "expected a numeric matrix, got {}",
                other.type_name()
            ))),
        }
    }

    pub fn real_scalar(&self) -> MatResult<f64> {
        match self {
            Value::Matrix(m) => m.real_scalar(),
            Value::Fi(fi) if fi.rows * fi.cols == 1 => Ok(fi.to_f64(fi.raw[0])),
            _ => Err(MatError::new("expected a real scalar")),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Matrix(m) if m.logical => "logical",
            Value::Matrix(_) => "double",
            Value::Char(_) => "char",
            Value::Cell { .. } => "cell",
            Value::Struct(_) => "struct",
            Value::Func(_) => "function_handle",
            Value::Fi(_) => "fi",
            Value::System(System::Tf { .. }) => "tf",
            Value::System(System::Ss { .. }) => "ss",
            Value::Multi(_) => "multi",
        }
    }

    pub fn summary(&self) -> String {
        match self {
            Value::Matrix(m) if m.logical => format!("{}x{} logical", m.rows, m.cols),
            Value::Matrix(m) if !m.is_real() => format!("{}x{} complex", m.rows, m.cols),
            Value::Matrix(m) => format!("{}x{} double", m.rows, m.cols),
            Value::Char(s) => format!("1x{} char", s.chars().count()),
            Value::Cell { rows, cols, .. } => format!("{rows}x{cols} cell"),
            Value::Struct(map) => format!("struct with {} fields", map.len()),
            Value::Func(_) => "function_handle".to_string(),
            Value::Fi(fi) => format!(
                "{}x{} fi {}{},{}",
                fi.rows,
                fi.cols,
                if fi.signed { "s" } else { "u" },
                fi.word,
                fi.frac
            ),
            Value::System(System::Tf { .. }) => "tf".to_string(),
            Value::System(System::Ss { .. }) => "ss".to_string(),
            Value::Multi(v) => format!("{} outputs", v.len()),
        }
    }

    pub fn is_true(&self) -> MatResult<bool> {
        let s = self.real_scalar()?;
        Ok(s != 0.0 && !s.is_nan())
    }
}

fn broadcast_len(a: usize, b: usize) -> Result<usize, ()> {
    if a == b || a == 1 {
        Ok(b)
    } else if b == 1 {
        Ok(a)
    } else {
        Err(())
    }
}

pub fn nearly_equal(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= EPS * (1.0 + a.abs().max(b.abs()))
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Matrix(m) => write!(f, "{}", format_matrix(m)),
            Value::Char(s) => write!(f, "'{s}'"),
            Value::Cell { rows, cols, data } => {
                write!(f, "{rows}x{cols} cell")?;
                for (i, v) in data.iter().enumerate() {
                    write!(f, "\n  {{{}}} {}", i + 1, v.summary())?;
                }
                Ok(())
            }
            Value::Struct(map) => {
                write!(f, "struct")?;
                for (k, v) in map {
                    write!(f, "\n  {k}: {}", v.summary())?;
                }
                Ok(())
            }
            Value::Func(FuncHandle::Named(n)) => write!(f, "@{n}"),
            Value::Func(FuncHandle::Anon { params, .. }) => {
                write!(f, "@({})", params.join(", "))
            }
            Value::Fi(fi) => {
                write!(
                    f,
                    "fi {}{},{}\n{}",
                    if fi.signed { "s" } else { "u" },
                    fi.word,
                    fi.frac,
                    format_matrix(&fi.to_matrix())
                )
            }
            Value::System(System::Tf { num, den }) => {
                write!(f, "tf num=[{}] den=[{}]", fmt_vec(num), fmt_vec(den))
            }
            Value::System(System::Ss { a, .. }) => {
                write!(f, "ss {}x{}", a.rows, a.cols)
            }
            Value::Multi(v) => {
                for (i, item) in v.iter().enumerate() {
                    if i > 0 {
                        writeln!(f)?;
                    }
                    write!(f, "{item}")?;
                }
                Ok(())
            }
        }
    }
}

fn fmt_vec(v: &[f64]) -> String {
    v.iter()
        .map(|x| format_number(*x))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn format_number(v: f64) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    if (v - v.round()).abs() < 1e-10 && v.abs() < 1e12 {
        format!("{}", v.round() as i64)
    } else {
        let s = format!("{v:.6}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

pub fn format_matrix(m: &Matrix) -> String {
    if m.numel() == 0 {
        return "[]".to_string();
    }
    let mut cols_txt: Vec<Vec<String>> = Vec::new();
    for c in 0..m.cols {
        let mut col = Vec::new();
        for r in 0..m.rows {
            let z = m.get(r, c);
            let text = if m.logical {
                if z.re != 0.0 { "1".to_string() } else { "0".to_string() }
            } else if z.im.abs() < 1e-12 {
                format_number(z.re)
            } else if z.im >= 0.0 {
                format!("{}+{}i", format_number(z.re), format_number(z.im))
            } else {
                format!("{}{}i", format_number(z.re), format_number(z.im))
            };
            col.push(text);
        }
        cols_txt.push(col);
    }
    let widths: Vec<usize> = cols_txt
        .iter()
        .map(|col| col.iter().map(|s| s.len()).max().unwrap_or(0))
        .collect();
    let mut lines = Vec::new();
    for r in 0..m.rows {
        let mut parts = Vec::new();
        for c in 0..m.cols {
            parts.push(format!("{:>width$}", cols_txt[c][r], width = widths[c].max(1)));
        }
        lines.push(parts.join("  "));
    }
    lines.join("\n")
}

pub fn range_values(start: f64, step: f64, end: f64) -> MatResult<Vec<f64>> {
    if step == 0.0 || !step.is_finite() {
        return Err(MatError::new("colon step must be a non-zero finite number"));
    }
    if (end - start).signum() != step.signum() && (end - start).abs() > 1e-12 {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut v = start;
    let max_n = 1_000_000usize;
    if step > 0.0 {
        while v <= end + 1e-9 * (1.0 + end.abs()) {
            out.push(v);
            v += step;
            if out.len() > max_n {
                return Err(MatError::new("colon range is too long"));
            }
        }
    } else {
        while v >= end - 1e-9 * (1.0 + end.abs()) {
            out.push(v);
            v += step;
            if out.len() > max_n {
                return Err(MatError::new("colon range is too long"));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementwise_product_expands_a_row_and_a_column() {
        let a = Matrix::from_reals(1, 3, vec![2.0, 4.0, 8.0]);
        let b = Matrix::from_reals(3, 1, vec![1.0, 2.0, 3.0]);
        let c = a.ew(&b, |x, y| x * y).unwrap();
        assert_eq!((c.rows, c.cols), (3, 3));
        assert_eq!(
            c.reals().unwrap(),
            vec![2.0, 4.0, 6.0, 4.0, 8.0, 12.0, 8.0, 16.0, 24.0]
        );
    }
}
