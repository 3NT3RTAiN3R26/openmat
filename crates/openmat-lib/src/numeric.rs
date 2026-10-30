use nalgebra::linalg::{Schur, SVD, SymmetricEigen};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Matrix, Session, Value};

use crate::util::{col, dim_pair, matrix_arg, need, positive, real_scalar, row};

pub fn register(session: &mut Session) {
    session.ldiv_hook = Some(solve);
    session.rdiv_hook = Some(right_solve);
    for (name, func) in [
        ("zeros", zeros as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("ones", ones),
        ("eye", eye),
        ("size", size),
        ("length", length),
        ("numel", numel),
        ("ndims", ndims),
        ("isempty", isempty),
        ("diag", diag),
        ("linspace", linspace),
        ("reshape", reshape),
        ("sum", sum),
        ("mean", mean),
        ("min", min),
        ("max", max),
        ("inv", inv),
        ("det", det),
        ("eig", eig),
        ("svd", svd),
        ("rand", rand),
        ("randn", randn),
        ("rng", rng),
        ("abs", abs),
        ("sqrt", sqrt),
        ("sin", sin),
        ("cos", cos),
        ("tan", tan),
        ("exp", exp),
        ("log", log),
        ("log10", log10),
        ("floor", floor),
        ("ceil", ceil),
        ("round", round),
        ("sign", sign),
        ("real", real),
        ("imag", imag),
        ("conj", conj),
        ("transpose", transpose),
        ("ctranspose", ctranspose),
    ] {
        session.register_builtin(name, func);
    }
}

fn zeros(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (r, c) = dim_pair(args)?;
    Ok(Value::Matrix(Matrix::zeros(r, c)))
}

fn ones(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (r, c) = dim_pair(args)?;
    Ok(Value::Matrix(Matrix::from_reals(r, c, vec![1.0; r * c])))
}

fn eye(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (r, c) = dim_pair(args)?;
    let mut m = Matrix::zeros(r, c);
    for i in 0..r.min(c) {
        m.set(i, i, Complex64::from(1.0));
    }
    Ok(Value::Matrix(m))
}

fn size(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let m = matrix_arg(args, 0)?;
    if args.len() >= 2 {
        let dim = positive(real_scalar(args, 1)?)?;
        let n = if dim == 1 { m.rows } else { m.cols };
        return Ok(Value::scalar(n as f64));
    }
    if session.nargout >= 2 {
        return Ok(Value::Multi(vec![
            Value::scalar(m.rows as f64),
            Value::scalar(m.cols as f64),
        ]));
    }
    Ok(row(vec![m.rows as f64, m.cols as f64]))
}

fn length(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    Ok(Value::scalar(m.rows.max(m.cols) as f64))
}

fn numel(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    Ok(Value::scalar(m.numel() as f64))
}

fn ndims(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let _ = matrix_arg(args, 0)?;
    Ok(Value::scalar(2.0))
}

fn isempty(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    Ok(Value::logical(m.numel() == 0))
}

fn diag(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    if m.rows == 1 || m.cols == 1 {
        let n = m.numel();
        let mut out = Matrix::zeros(n, n);
        for i in 0..n {
            let r = i % m.rows;
            let c = i / m.rows;
            out.set(i, i, m.get(r, c));
        }
        return Ok(Value::Matrix(out));
    }
    let n = m.rows.min(m.cols);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push(m.get(i, i).re);
    }
    Ok(col(data))
}

fn linspace(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let a = real_scalar(args, 0)?;
    let b = real_scalar(args, 1)?;
    let n = if args.len() >= 3 {
        positive(real_scalar(args, 2)?)?
    } else {
        100
    };
    if n == 0 {
        return Ok(row(Vec::new()));
    }
    if n == 1 {
        return Ok(row(vec![b]));
    }
    let step = (b - a) / (n - 1) as f64;
    Ok(row((0..n).map(|i| a + step * i as f64).collect()))
}

fn reshape(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let m = matrix_arg(args, 0)?;
    let rows = positive(real_scalar(args, 1)?)?;
    let cols = positive(real_scalar(args, 2)?)?;
    if rows * cols != m.numel() {
        return Err(MatError::new("reshape element count must stay the same"));
    }
    Ok(Value::Matrix(Matrix {
        rows,
        cols,
        data: m.data,
        logical: m.logical,
    }))
}

fn sum(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    let dim = if args.len() >= 2 {
        positive(real_scalar(args, 1)?)?
    } else if m.rows == 1 {
        2
    } else {
        1
    };
    reduce(&m, dim, 0.0, |a, b| a + b)
}

fn mean(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    let dim = if args.len() >= 2 {
        positive(real_scalar(args, 1)?)?
    } else if m.rows == 1 {
        2
    } else {
        1
    };
    let total = reduce(&m, dim, 0.0, |a, b| a + b)?.into_matrix()?;
    let n = if dim == 1 { m.rows } else { m.cols } as f64;
    Ok(Value::Matrix(total.map(|z| z / n)))
}

fn min(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    extremum(session, args, |a, b| a < b)
}

fn max(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    extremum(session, args, |a, b| a > b)
}

fn extremum(_: &mut Session, args: &[Value], better: impl Fn(f64, f64) -> bool) -> MatResult<Value> {
    need(args, 1)?;
    if args.len() >= 2 {
        let a = matrix_arg(args, 0)?;
        let b = matrix_arg(args, 1)?;
        return Ok(Value::Matrix(a.ew(&b, |x, y| {
            if better(x.re, y.re) {
                x
            } else {
                y
            }
        })?));
    }
    let m = matrix_arg(args, 0)?;
    if m.rows == 1 || m.cols == 1 {
        let v = m
            .reals()?
            .into_iter()
            .reduce(|a, b| if better(a, b) { a } else { b })
            .unwrap_or(f64::NAN);
        return Ok(Value::scalar(v));
    }
    let mut data = Vec::new();
    for c in 0..m.cols {
        let mut best = m.get(0, c).re;
        for r in 1..m.rows {
            let v = m.get(r, c).re;
            if better(v, best) {
                best = v;
            }
        }
        data.push(best);
    }
    Ok(row(data))
}

fn map_real(args: &[Value], f: impl Fn(f64) -> Complex64) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    Ok(Value::Matrix(m.map(|z| f(z.re))))
}

fn abs(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    Ok(Value::Matrix(m.map(|z| Complex64::from(z.norm()))))
}

fn sqrt(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    Ok(Value::Matrix(m.map(|z| z.sqrt())))
}

fn sin(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    map_real(args, |v| Complex64::from(v.sin()))
}
fn cos(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    map_real(args, |v| Complex64::from(v.cos()))
}
fn tan(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    map_real(args, |v| Complex64::from(v.tan()))
}
fn exp(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    Ok(Value::Matrix(m.map(|z| z.exp())))
}
fn log(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    Ok(Value::Matrix(m.map(|z| z.ln())))
}
fn log10(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    map_real(args, |v| Complex64::from(v.log10()))
}
fn floor(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    map_real(args, |v| Complex64::from(v.floor()))
}
fn ceil(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    map_real(args, |v| Complex64::from(v.ceil()))
}
fn round(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    map_real(args, |v| Complex64::from(v.round()))
}
fn sign(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    map_real(args, |v| Complex64::from(v.signum()))
}
fn real(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    Ok(Value::Matrix(m.map(|z| Complex64::from(z.re))))
}
fn imag(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    Ok(Value::Matrix(m.map(|z| Complex64::from(z.im))))
}
fn conj(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    Ok(Value::Matrix(m.map(|z| z.conj())))
}
fn transpose(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    Ok(Value::Matrix(matrix_arg(args, 0)?.transpose(false)))
}
fn ctranspose(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    Ok(Value::Matrix(matrix_arg(args, 0)?.transpose(true)))
}

fn rand(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (r, c) = if args.is_empty() { (1, 1) } else { dim_pair(args)? };
    let mut data = Vec::with_capacity(r * c);
    for _ in 0..r * c {
        data.push(session.rand_f64());
    }
    Ok(Value::Matrix(Matrix::from_reals(r, c, data)))
}

fn randn(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (r, c) = if args.is_empty() { (1, 1) } else { dim_pair(args)? };
    let mut data = Vec::with_capacity(r * c);
    while data.len() < r * c {
        let u1 = session.rand_f64().clamp(1e-12, 1.0);
        let u2 = session.rand_f64();
        let mag = (-2.0 * u1.ln()).sqrt();
        data.push(mag * (2.0 * std::f64::consts::PI * u2).cos());
        if data.len() < r * c {
            data.push(mag * (2.0 * std::f64::consts::PI * u2).sin());
        }
    }
    Ok(Value::Matrix(Matrix::from_reals(r, c, data)))
}

fn rng(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let seed = real_scalar(args, 0)?.to_bits().max(1);
    session.rng = seed;
    Ok(crate::util::empty())
}

fn inv(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    let dm = to_dmatrix(&m)?;
    let inv = dm
        .try_inverse()
        .ok_or_else(|| MatError::new("matrix is singular to working precision"))?;
    Ok(Value::Matrix(from_dmatrix(&inv)))
}

fn det(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    let dm = to_dmatrix(&m)?;
    Ok(Value::scalar(dm.determinant()))
}

fn eig(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    if m.rows != m.cols {
        return Err(MatError::new("eig needs a square matrix"));
    }
    let dm = to_dmatrix(&m)?;
    if session.nargout >= 2 {
        if !is_symmetric(&m) {
            return Err(MatError::new(
                "two-output eig is implemented for real symmetric matrices",
            ));
        }
        let se = SymmetricEigen::new(dm);
        let vectors = from_dmatrix(&se.eigenvectors);
        let values = Matrix::from_reals(m.rows, 1, se.eigenvalues.iter().copied().collect());
        let mut diag = Matrix::zeros(m.rows, m.rows);
        for i in 0..m.rows {
            diag.set(i, i, Complex64::from(values.get(i, 0).re));
        }
        return Ok(Value::Multi(vec![Value::Matrix(vectors), Value::Matrix(diag)]));
    }
    let values = complex_eigenvalues(&dm)?;
    let mut out = Matrix::zeros(values.len(), 1);
    for (i, v) in values.iter().enumerate() {
        out.set(i, 0, *v);
    }
    Ok(Value::Matrix(out))
}

fn svd(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    let dm = to_dmatrix(&m)?;
    let svd = SVD::new(dm, true, true);
    if session.nargout >= 3 {
        let u = from_dmatrix(&svd.u.expect("u"));
        let vt = from_dmatrix(&svd.v_t.expect("v_t"));
        let mut s = Matrix::zeros(m.rows, m.cols);
        for (i, value) in svd.singular_values.iter().enumerate() {
            if i < m.rows && i < m.cols {
                s.set(i, i, Complex64::from(*value));
            }
        }
        return Ok(Value::Multi(vec![
            Value::Matrix(u),
            Value::Matrix(s),
            Value::Matrix(vt.transpose(false)),
        ]));
    }
    Ok(col(svd.singular_values.iter().copied().collect()))
}

pub fn solve(a: &Matrix, b: &Matrix) -> MatResult<Matrix> {
    let da = to_dmatrix(a)?;
    let db = to_dmatrix(b)?;
    if a.rows == a.cols {
        let lu = da.lu();
        let mut cols = Vec::new();
        for c in 0..b.cols {
            let sol = lu
                .solve(&db.column(c).into_owned())
                .ok_or_else(|| MatError::new("matrix is singular to working precision"))?;
            cols.push(sol);
        }
        return Ok(hcat_vectors(a.cols, cols));
    }
    let svd = SVD::new(da, true, true);
    let mut cols = Vec::new();
    for c in 0..b.cols {
        let sol = svd
            .solve(&db.column(c).into_owned(), 1e-12)
            .map_err(|err| MatError::new(err))?;
        cols.push(sol);
    }
    Ok(hcat_vectors(a.cols, cols))
}

fn right_solve(a: &Matrix, b: &Matrix) -> MatResult<Matrix> {
    let x = solve(&b.transpose(true), &a.transpose(true))?;
    Ok(x.transpose(true))
}

fn hcat_vectors(rows: usize, cols: Vec<DVector<f64>>) -> Matrix {
    let width = cols.len();
    let mut data = vec![0.0; rows * width];
    for (c, col) in cols.iter().enumerate() {
        for r in 0..rows {
            data[c * rows + r] = col[r];
        }
    }
    Matrix::from_reals(rows, width, data)
}

pub fn complex_eigenvalues(dm: &DMatrix<f64>) -> MatResult<Vec<Complex64>> {
    let schur = Schur::try_new(dm.clone(), 1e-12, 0)
        .ok_or_else(|| MatError::new("eig did not converge"))?;
    let values = schur.complex_eigenvalues();
    Ok(values.iter().copied().collect())
}

fn is_symmetric(m: &Matrix) -> bool {
    if m.rows != m.cols || !m.is_real() {
        return false;
    }
    for r in 0..m.rows {
        for c in 0..r {
            if (m.get(r, c).re - m.get(c, r).re).abs() > 1e-8 {
                return false;
            }
        }
    }
    true
}

pub fn to_dmatrix(m: &Matrix) -> MatResult<DMatrix<f64>> {
    let reals = m.reals()?;
    Ok(DMatrix::from_column_slice(m.rows, m.cols, &reals))
}

pub fn from_dmatrix(m: &DMatrix<f64>) -> Matrix {
    Matrix::from_reals(m.nrows(), m.ncols(), m.as_slice().to_vec())
}

fn reduce(m: &Matrix, dim: usize, init: f64, fold: impl Fn(f64, f64) -> f64) -> MatResult<Value> {
    if dim == 1 {
        let mut data = Vec::new();
        for c in 0..m.cols {
            let mut acc = init;
            for r in 0..m.rows {
                acc = fold(acc, m.get(r, c).re);
            }
            data.push(acc);
        }
        Ok(row(data))
    } else {
        let mut data = Vec::new();
        for r in 0..m.rows {
            let mut acc = init;
            for c in 0..m.cols {
                acc = fold(acc, m.get(r, c).re);
            }
            data.push(acc);
        }
        Ok(col(data))
    }
}
