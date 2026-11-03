use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Matrix, Session, Value};
use rustfft::FftPlanner;

use crate::numeric::{complex_eigenvalues, to_dmatrix};
use crate::util::{col, matrix_arg, need, positive, real_scalar, row};

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("rectwin", rectwin as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("bartlett", bartlett),
        ("hann", hann),
        ("hamming", hamming),
        ("blackman", blackman),
        ("kaiser", kaiser),
        ("gausswin", gausswin),
        ("tukeywin", tukeywin),
        ("nextpow2", nextpow2),
        ("fftshift", fftshift),
        ("ifftshift", ifftshift),
        ("sinc", sinc),
        ("angle", angle),
        ("cumsum", cumsum),
        ("diff", diff),
        ("unwrap", unwrap),
        ("deconv", deconv),
        ("xcorr", xcorr),
        ("polyval", polyval),
        ("poly", poly),
        ("roots", roots),
        ("hilbert", hilbert),
        ("detrend", detrend),
    ] {
        session.register_builtin(name, func);
    }
}

// Windows

/// Window length and the `'periodic'` or `'symmetric'` flag from `(L, ..., flag)`.
fn window_len(args: &[Value]) -> MatResult<(usize, bool)> {
    need(args, 1)?;
    let n = positive(real_scalar(args, 0)?)?;
    let periodic = match args.last() {
        Some(Value::Char(flag)) if flag == "periodic" => true,
        Some(Value::Char(flag)) if flag == "symmetric" => false,
        Some(Value::Char(other)) => {
            return Err(MatError::new(format!(
                "window type must be 'symmetric' or 'periodic', got '{other}'"
            )))
        }
        _ => false,
    };
    Ok((n, periodic))
}

/// `sum_k coeffs[k] * cos(2 pi k i / denominator)` for a symmetric or periodic window of length `n`.
fn cosine_window(args: &[Value], coeffs: &[f64]) -> MatResult<Value> {
    let (n, periodic) = window_len(args)?;
    if n <= 1 {
        return Ok(col(vec![1.0; n]));
    }
    let denominator = if periodic { n as f64 } else { (n - 1) as f64 };
    let window = (0..n)
        .map(|i| {
            coeffs
                .iter()
                .enumerate()
                .map(|(k, a)| a * (2.0 * PI * k as f64 * i as f64 / denominator).cos())
                .sum()
        })
        .collect();
    Ok(col(window))
}

fn rectwin(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    Ok(col(vec![1.0; positive(real_scalar(args, 0)?)?]))
}

fn bartlett(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    if args.len() > 1 {
        return Err(MatError::new("bartlett takes only the window length"));
    }
    let (n, _) = window_len(args)?;
    if n <= 1 {
        return Ok(col(vec![1.0; n]));
    }
    let denominator = (n - 1) as f64;
    Ok(col(
        (0..n)
            .map(|i| 1.0 - (2.0 * i as f64 / denominator - 1.0).abs())
            .collect(),
    ))
}

fn hann(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    cosine_window(args, &[0.5, -0.5])
}

fn hamming(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    cosine_window(args, &[0.54, -0.46])
}

fn blackman(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    cosine_window(args, &[0.42, -0.5, 0.08])
}

/// Modified Bessel function of the first kind, order zero.
pub(crate) fn bessel_i0(x: f64) -> f64 {
    let quarter = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..500 {
        term *= quarter / (k as f64 * k as f64);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

fn kaiser(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let n = positive(real_scalar(args, 0)?)?;
    let beta = if args.len() >= 2 { real_scalar(args, 1)? } else { 0.5 };
    if n <= 1 {
        return Ok(col(vec![1.0; n]));
    }
    let norm = bessel_i0(beta);
    Ok(col(
        (0..n)
            .map(|i| {
                let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
                bessel_i0(beta * (1.0 - x * x).max(0.0).sqrt()) / norm
            })
            .collect(),
    ))
}

fn gausswin(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let n = positive(real_scalar(args, 0)?)?;
    let alpha = if args.len() >= 2 { real_scalar(args, 1)? } else { 2.5 };
    if n <= 1 {
        return Ok(col(vec![1.0; n]));
    }
    let half = (n - 1) as f64 / 2.0;
    Ok(col(
        (0..n)
            .map(|i| {
                let t = alpha * (i as f64 - half) / half;
                (-0.5 * t * t).exp()
            })
            .collect(),
    ))
}

fn tukeywin(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let n = positive(real_scalar(args, 0)?)?;
    let r = if args.len() >= 2 { real_scalar(args, 1)? } else { 0.5 };
    if n <= 1 {
        return Ok(col(vec![1.0; n]));
    }
    let r = r.clamp(0.0, 1.0);
    if r == 0.0 {
        return Ok(col(vec![1.0; n]));
    }
    Ok(col(
        (0..n)
            .map(|i| {
                let x = i as f64 / (n - 1) as f64;
                if x < r / 2.0 {
                    0.5 * (1.0 + (2.0 * PI / r * (x - r / 2.0)).cos())
                } else if x >= 1.0 - r / 2.0 {
                    0.5 * (1.0 + (2.0 * PI / r * (x - 1.0 + r / 2.0)).cos())
                } else {
                    1.0
                }
            })
            .collect(),
    ))
}

// Elementwise helpers

fn nextpow2(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    Ok(Value::Matrix(m.map(|z| {
        let magnitude = z.norm();
        Complex64::from(if magnitude == 0.0 {
            0.0
        } else {
            let exponent = magnitude.log2().ceil();
            if exponent.exp2() < magnitude {
                exponent + 1.0
            } else {
                exponent
            }
        })
    })))
}

fn sinc(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    Ok(Value::Matrix(m.map(|z| {
        if z == Complex64::new(0.0, 0.0) {
            Complex64::new(1.0, 0.0)
        } else {
            let x = z * PI;
            x.sin() / x
        }
    })))
}

fn angle(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    Ok(Value::Matrix(m.map(|z| Complex64::from(z.arg()))))
}

// Work along a dimension

pub(crate) fn default_dim(m: &Matrix) -> usize {
    if m.rows == 1 {
        2
    } else {
        1
    }
}

fn dim_arg(args: &[Value], index: usize, m: &Matrix) -> MatResult<usize> {
    if args.len() > index {
        let dim = positive(real_scalar(args, index)?)?;
        if dim != 1 && dim != 2 {
            return Err(MatError::new("dimension must be 1 or 2"));
        }
        Ok(dim)
    } else {
        Ok(default_dim(m))
    }
}

/// Applies `f` to every column (`dim` 1) or row (`dim` 2). `f` may change the length.
pub(crate) fn along(
    m: &Matrix,
    dim: usize,
    empty_rows: usize,
    f: impl Fn(&[Complex64]) -> Vec<Complex64>,
) -> MatResult<Matrix> {
    if dim == 2 {
        return Ok(along(&m.transpose(false), 1, empty_rows, f)?.transpose(false));
    }
    let mut data = Vec::new();
    let mut rows = empty_rows;
    for c in 0..m.cols {
        let out = f(&m.data[c * m.rows..(c + 1) * m.rows]);
        rows = out.len();
        data.extend(out);
    }
    Ok(Matrix {
        rows,
        cols: m.cols,
        data,
        logical: false,
    })
}

fn cumsum(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    let dim = dim_arg(args, 1, &m)?;
    let rows = if dim == 1 { m.rows } else { m.cols };
    Ok(Value::Matrix(along(&m, dim, rows, |x| {
        let mut total = Complex64::new(0.0, 0.0);
        x.iter()
            .map(|v| {
                total += v;
                total
            })
            .collect()
    })?))
}

fn diff(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let mut m = matrix_arg(args, 0)?;
    let order = if args.len() >= 2 { positive(real_scalar(args, 1)?)? } else { 1 };
    let dim = dim_arg(args, 2, &m)?;
    for _ in 0..order {
        let rows = if dim == 1 { m.rows } else { m.cols }.saturating_sub(1);
        m = along(&m, dim, rows, |x| {
            x.windows(2).map(|w| w[1] - w[0]).collect()
        })?;
    }
    Ok(Value::Matrix(m))
}

fn shift_matrix(m: &Matrix, dim: Option<usize>, inverse: bool) -> MatResult<Matrix> {
    let amount = |n: usize| if inverse { n - n / 2 } else { n / 2 };
    let (shift_rows, shift_cols) = match dim {
        None if m.rows == 1 => (false, true),
        None if m.cols == 1 => (true, false),
        None => (true, true),
        Some(1) => (true, false),
        Some(2) => (false, true),
        Some(_) => return Err(MatError::new("dimension must be 1 or 2")),
    };
    let shift_by_rows = if shift_rows { amount(m.rows) } else { 0 };
    let shift_by_cols = if shift_cols { amount(m.cols) } else { 0 };
    let mut out = m.clone();
    for r in 0..m.rows {
        for c in 0..m.cols {
            out.set((r + shift_by_rows) % m.rows, (c + shift_by_cols) % m.cols, m.get(r, c));
        }
    }
    Ok(out)
}

fn shift_builtin(args: &[Value], inverse: bool) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    let dim = if args.len() >= 2 {
        Some(positive(real_scalar(args, 1)?)?)
    } else {
        None
    };
    Ok(Value::Matrix(shift_matrix(&m, dim, inverse)?))
}

fn fftshift(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    shift_builtin(args, false)
}

fn ifftshift(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    shift_builtin(args, true)
}

fn unwrap_reals(p: &[f64], cutoff: f64) -> Vec<f64> {
    let mut out = p.to_vec();
    let mut offset = 0.0;
    for i in 1..p.len() {
        let step = p[i] - p[i - 1];
        let mut wrapped = (step + PI).rem_euclid(2.0 * PI) - PI;
        if wrapped == -PI && step > 0.0 {
            wrapped = PI;
        }
        if step.abs() >= cutoff {
            offset += wrapped - step;
        }
        out[i] = p[i] + offset;
    }
    out
}

fn unwrap(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    let cutoff = if args.len() >= 2 { real_scalar(args, 1)? } else { PI };
    let dim = dim_arg(args, 2, &m)?;
    let rows = if dim == 1 { m.rows } else { m.cols };
    Ok(Value::Matrix(along(&m, dim, rows, |x| {
        let reals: Vec<f64> = x.iter().map(|z| z.re).collect();
        unwrap_reals(&reals, cutoff)
            .into_iter()
            .map(Complex64::from)
            .collect()
    })?))
}

// Polynomials

fn deconv(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let y_matrix = matrix_arg(args, 0)?;
    let y = y_matrix.reals()?;
    let a = matrix_arg(args, 1)?.reals()?;
    if a.is_empty() || a[0] == 0.0 {
        return Err(MatError::new("deconv divisor a(1) must be non-zero"));
    }
    // The outputs are columns when the dividend is a column vector, otherwise rows.
    let column = y_matrix.cols == 1 && y_matrix.rows > 1;
    let oriented = |values: Vec<f64>| if column { col(values) } else { row(values) };
    let (quotient, remainder) = if y.len() < a.len() {
        (vec![0.0], y)
    } else {
        let mut remainder = y;
        let mut quotient = vec![0.0; remainder.len() - a.len() + 1];
        for i in 0..quotient.len() {
            quotient[i] = remainder[i] / a[0];
            for (j, aj) in a.iter().enumerate() {
                remainder[i + j] -= quotient[i] * aj;
            }
            remainder[i] = 0.0;
        }
        (quotient, remainder)
    };
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![oriented(quotient), oriented(remainder)]))
    } else {
        Ok(oriented(quotient))
    }
}

fn polyval(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let p = matrix_arg(args, 0)?;
    let x = matrix_arg(args, 1)?;
    Ok(Value::Matrix(x.map(|z| {
        p.data
            .iter()
            .fold(Complex64::new(0.0, 0.0), |acc, c| acc * z + c)
    })))
}

fn poly(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    let zeros: Vec<Complex64> = if m.rows == 1 || m.cols == 1 {
        m.data.clone()
    } else if m.rows == m.cols {
        complex_eigenvalues(&to_dmatrix(&m)?)?
    } else {
        return Err(MatError::new("poly needs a vector or a square matrix"));
    };
    Ok(coefficient_row(poly_from_roots(&zeros), conjugate_closed(&zeros)))
}

/// Coefficients, highest power first, of the monic polynomial with the given roots.
pub(crate) fn poly_from_roots(zeros: &[Complex64]) -> Vec<Complex64> {
    let mut coeff = vec![Complex64::new(1.0, 0.0)];
    for z in zeros {
        let mut next = vec![Complex64::new(0.0, 0.0); coeff.len() + 1];
        for (i, c) in coeff.iter().enumerate() {
            next[i] += c;
            next[i + 1] -= c * z;
        }
        coeff = next;
    }
    coeff
}

/// True when every non-real root has its complex conjugate among the roots.
pub(crate) fn conjugate_closed(roots: &[Complex64]) -> bool {
    let mut unmatched: Vec<Complex64> = roots.iter().copied().filter(|r| r.im != 0.0).collect();
    while let Some(root) = unmatched.pop() {
        let close = |other: &Complex64| {
            let tolerance = 1e-9 * (1.0 + root.norm());
            (other.re - root.re).abs() <= tolerance && (other.im + root.im).abs() <= tolerance
        };
        match unmatched.iter().position(close) {
            Some(partner) => {
                unmatched.swap_remove(partner);
            }
            None => return false,
        }
    }
    true
}

/// A row vector of the coefficients. `real` drops the imaginary parts, which are then rounding noise.
pub(crate) fn coefficient_row(coeff: Vec<Complex64>, real: bool) -> Value {
    if real {
        return row(coeff.iter().map(|c| c.re).collect());
    }
    Value::Matrix(Matrix {
        rows: 1,
        cols: coeff.len(),
        data: coeff,
        logical: false,
    })
}

fn roots(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    if m.rows != 1 && m.cols != 1 && m.numel() != 0 {
        return Err(MatError::new("roots needs a vector of coefficients"));
    }
    let found = polynomial_roots(&m.reals()?)?;
    Ok(Value::Matrix(Matrix {
        rows: found.len(),
        cols: 1,
        data: found,
        logical: false,
    }))
}

/// Roots of a real polynomial. Leading zeros are dropped, trailing zeros give roots at 0.
pub(crate) fn polynomial_roots(coefficients: &[f64]) -> MatResult<Vec<Complex64>> {
    let mut p = coefficients.to_vec();
    let leading = p.iter().take_while(|c| **c == 0.0).count();
    p.drain(..leading);
    let trailing = p.iter().rev().take_while(|c| **c == 0.0).count();
    p.truncate(p.len() - trailing);
    let mut found: Vec<Complex64> = match p.len() {
        0 | 1 => Vec::new(),
        2 => vec![Complex64::from(-p[1] / p[0])],
        n => {
            let degree = n - 1;
            let mut companion = DMatrix::<f64>::zeros(degree, degree);
            for c in 0..degree {
                companion[(0, c)] = -p[c + 1] / p[0];
            }
            for r in 1..degree {
                companion[(r, r - 1)] = 1.0;
            }
            complex_eigenvalues(&companion)?
        }
    };
    found.extend(std::iter::repeat_n(Complex64::new(0.0, 0.0), trailing));
    // MATLAB lists the member with the positive imaginary part first.
    for i in 0..found.len().saturating_sub(1) {
        if found[i].im < 0.0 && found[i + 1] == found[i].conj() {
            found.swap(i, i + 1);
        }
    }
    Ok(found)
}

// Analysis

fn analytic_signal(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mut buf: Vec<Complex64> = x.iter().map(|z| Complex64::from(z.re)).collect();
    let mut planner = FftPlanner::<f64>::new();
    planner.plan_fft_forward(n).process(&mut buf);
    for (k, v) in buf.iter_mut().enumerate() {
        let gain = if k == 0 || (n.is_multiple_of(2) && k == n / 2) {
            1.0
        } else if k < n.div_ceil(2) {
            2.0
        } else {
            0.0
        };
        *v *= gain;
    }
    planner.plan_fft_inverse(n).process(&mut buf);
    for v in &mut buf {
        *v /= n as f64;
    }
    buf
}

fn hilbert(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    let dim = default_dim(&m);
    let rows = if dim == 1 { m.rows } else { m.cols };
    Ok(Value::Matrix(along(&m, dim, rows, analytic_signal)?))
}

fn detrend_reals(y: &[f64], order: usize) -> MatResult<Vec<f64>> {
    let n = y.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if order == 0 {
        let mean = y.iter().sum::<f64>() / n as f64;
        return Ok(y.iter().map(|v| v - mean).collect());
    }
    // Fit on t in [-1, 1] to keep the Vandermonde matrix well conditioned.
    let design = DMatrix::from_fn(n, order + 1, |i, j| {
        let t = if n > 1 { 2.0 * i as f64 / (n - 1) as f64 - 1.0 } else { 0.0 };
        t.powi(j as i32)
    });
    let target = DVector::from_column_slice(y);
    let solution = design
        .clone()
        .svd(true, true)
        .solve(&target, 1e-12)
        .map_err(|err| MatError::new(format!("detrend failed: {err}")))?;
    let fitted = design * solution;
    Ok(y.iter().zip(fitted.iter()).map(|(v, f)| v - f).collect())
}

fn detrend(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    let order = match args.get(1) {
        None => 1,
        Some(Value::Char(kind)) if kind == "constant" => 0,
        Some(Value::Char(kind)) if kind == "linear" => 1,
        Some(Value::Char(other)) => {
            return Err(MatError::new(format!(
                "detrend type must be 'constant' or 'linear', got '{other}'"
            )))
        }
        Some(value) => positive(value.real_scalar()?)?,
    };
    let dim = default_dim(&m);
    let rows = if dim == 1 { m.rows } else { m.cols };
    let result = along(&m, dim, rows, |x| {
        let re: Vec<f64> = x.iter().map(|z| z.re).collect();
        let im: Vec<f64> = x.iter().map(|z| z.im).collect();
        match (detrend_reals(&re, order), detrend_reals(&im, order)) {
            (Ok(re), Ok(im)) => re
                .into_iter()
                .zip(im)
                .map(|(r, i)| Complex64::new(r, i))
                .collect(),
            _ => vec![Complex64::new(f64::NAN, f64::NAN); x.len()],
        }
    })?;
    Ok(Value::Matrix(result))
}

/// Scaling option of `xcorr`.
enum XcorrScale {
    None,
    Biased,
    Unbiased,
    Coeff,
}

pub(crate) fn xcorr(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let mut scale = XcorrScale::None;
    let mut numeric: Vec<&Value> = Vec::new();
    for arg in args {
        match arg {
            Value::Char(option) => {
                scale = match option.as_str() {
                    "none" | "raw" => XcorrScale::None,
                    "biased" => XcorrScale::Biased,
                    "unbiased" => XcorrScale::Unbiased,
                    "coeff" | "normalized" => XcorrScale::Coeff,
                    other => {
                        return Err(MatError::new(format!("unknown xcorr scale option '{other}'")))
                    }
                }
            }
            other => numeric.push(other),
        }
    }
    if numeric.is_empty() {
        return Err(MatError::new("xcorr needs a numeric input vector"));
    }
    let x = numeric[0].clone().into_matrix()?;
    let (second, max_lag) = match numeric.len() {
        1 => (None, None),
        2 => {
            let second = numeric[1].clone().into_matrix()?;
            if second.numel() == 1 {
                (None, Some(second.real_scalar()?))
            } else {
                (Some(second), None)
            }
        }
        3 => (Some(numeric[1].clone().into_matrix()?), Some(numeric[2].real_scalar()?)),
        _ => return Err(MatError::new("xcorr takes at most x, y, maxlag and a scale option")),
    };
    if (x.rows != 1 && x.cols != 1) || second.as_ref().is_some_and(|y| y.rows != 1 && y.cols != 1) {
        return Err(MatError::new("xcorr supports vectors only"));
    }
    let y = second.as_ref().unwrap_or(&x);
    if x.numel() != y.numel() && !matches!(scale, XcorrScale::None) {
        return Err(MatError::new(
            "Scale option must be 'none' for input vectors of different lengths.",
        ));
    }
    let n = x.numel().max(y.numel());
    let pad = |v: &Matrix| {
        let mut data = v.data.clone();
        data.resize(n, Complex64::new(0.0, 0.0));
        data
    };
    let (xs, ys) = (pad(&x), pad(y));
    let max_lag = match max_lag {
        Some(lag) => positive(lag)?,
        None => n.saturating_sub(1),
    };
    let energy = |v: &[Complex64]| v.iter().map(|z| z.norm_sqr()).sum::<f64>();
    let coeff_norm = energy(&xs).sqrt() * energy(&ys).sqrt();
    let mut values = Vec::with_capacity(2 * max_lag + 1);
    let mut lags = Vec::with_capacity(2 * max_lag + 1);
    for lag in -(max_lag as i64)..=(max_lag as i64) {
        let mut total = Complex64::new(0.0, 0.0);
        for (i, yv) in ys.iter().enumerate() {
            let at = i as i64 + lag;
            if at >= 0 && (at as usize) < n {
                total += xs[at as usize] * yv.conj();
            }
        }
        let overlap = n as f64 - lag.unsigned_abs() as f64;
        total = match scale {
            XcorrScale::None => total,
            XcorrScale::Biased => total / n as f64,
            XcorrScale::Unbiased if overlap > 0.0 => total / overlap,
            XcorrScale::Unbiased => Complex64::new(0.0, 0.0),
            XcorrScale::Coeff if coeff_norm > 0.0 => total / coeff_norm,
            XcorrScale::Coeff => total,
        };
        values.push(total);
        lags.push(lag as f64);
    }
    let as_row = x.rows == 1 && x.cols > 1;
    let (rows, cols) = if as_row { (1, values.len()) } else { (values.len(), 1) };
    let c = Value::Matrix(Matrix {
        rows,
        cols,
        data: values,
        logical: false,
    });
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![c, row(lags)]))
    } else {
        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use crate::exec;

    fn reals(src: &str, name: &str) -> Vec<f64> {
        let session = exec(src).unwrap_or_else(|err| panic!("{src}: {err}"));
        session
            .get(name)
            .unwrap()
            .clone()
            .into_matrix()
            .unwrap()
            .reals()
            .unwrap()
    }

    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-9, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn windows_match_known_values() {
        assert_close(&reals("w = hann(4);", "w"), &[0.0, 0.75, 0.75, 0.0]);
        assert_close(&reals("w = hamming(3);", "w"), &[0.08, 1.0, 0.08]);
        assert_close(&reals("w = blackman(3);", "w"), &[0.0, 1.0, 0.0]);
        assert_close(&reals("w = bartlett(5);", "w"), &[0.0, 0.5, 1.0, 0.5, 0.0]);
        assert_close(&reals("w = rectwin(3);", "w"), &[1.0, 1.0, 1.0]);
        assert_close(&reals("w = hann(4, 'periodic');", "w"), &[0.0, 0.5, 1.0, 0.5]);
        assert_close(&reals("w = hann(1);", "w"), &[1.0]);
        assert!(reals("w = hann(0);", "w").is_empty());
        assert_close(&reals("w = kaiser(3, 0);", "w"), &[1.0, 1.0, 1.0]);
        let kaiser = reals("w = kaiser(5, 4);", "w");
        assert!((kaiser[2] - 1.0).abs() < 1e-12);
        assert!((kaiser[0] - 1.0 / 11.30192195213633).abs() < 1e-9);
        let gauss = reals("w = gausswin(3);", "w");
        assert!((gauss[0] - (-3.125f64).exp()).abs() < 1e-12 && gauss[1] == 1.0);
        let tukey = reals("w = tukeywin(9, 0.5);", "w");
        assert!((tukey[0]).abs() < 1e-12 && (tukey[1] - 0.5).abs() < 1e-12 && tukey[4] == 1.0);
        assert_close(&reals("w = tukeywin(4, 1);", "w"), &reals("w = hann(4);", "w"));
        assert!(crate::exec("w = hann(4, 'nope');").is_err());
    }

    #[test]
    fn matches_matlab_r2025a_reference_values() {
        assert_close(
            &reals("w = kaiser(7, 3.2);", "w"),
            &[0.173997555235272, 0.524660852056094, 0.861022193923715, 1.0, 0.861022193923715, 0.524660852056094, 0.173997555235272],
        );
        assert_close(
            &reals("w = gausswin(7, 2.1);", "w"),
            &[0.110250525304485, 0.375311098851399, 0.782704538241868, 1.0, 0.782704538241868, 0.375311098851399, 0.110250525304485],
        );
        assert_close(
            &reals("w = tukeywin(8, 0.3);", "w"),
            &[0.0, 0.994415413112564, 1.0, 1.0, 1.0, 1.0, 0.994415413112564, 0.0],
        );
        assert_close(&reals("w = tukeywin(7, 0.5);", "w"), &[0.0, 0.75, 1.0, 1.0, 1.0, 0.75, 0.0]);
        assert_close(
            &reals("w = hann(5, 'periodic');", "w"),
            &[0.0, 0.345491502812526, 0.904508497187474, 0.904508497187474, 0.345491502812526],
        );
        assert_close(&reals("w = hamming(6, 'periodic');", "w"), &[0.08, 0.31, 0.77, 1.0, 0.77, 0.31]);
        assert_close(
            &reals("w = blackman(6);", "w"),
            &[0.0, 0.20077014326253, 0.849229856737469, 0.849229856737469, 0.20077014326253, 0.0],
        );
        assert_close(&reals("w = bartlett(6);", "w"), &[0.0, 0.4, 0.8, 0.8, 0.4, 0.0]);
        assert!(crate::exec("w = bartlett(5, 'periodic');").is_err());
        assert_close(
            &reals("y = unwrap([0 4 8 12 -2 3]);", "y"),
            &[0.0, -2.28318530717959, -4.56637061435917, -6.84955592153876, -8.28318530717959, -9.56637061435917],
        );
        assert_close(
            &reals("y = detrend([1 2 4 8 7 3], 2);", "y"),
            &[1.10714285714286, -1.36428571428571, -1.51428571428571, 1.65714285714286, 1.15, -1.03571428571428],
        );
        assert_close(
            &reals("y = detrend([1 2 4 8 7 3], 'constant');", "y"),
            &[-3.16666666666667, -2.16666666666667, -0.166666666666667, 3.83333333333333, 2.83333333333333, -1.16666666666667],
        );
        for (src, want) in [
            (
                "z = imag(hilbert([1 3 2 5 4]));",
                vec![0.179611190631834, -0.470228201833978, -0.795147898066885, -1.52169042607225, 2.60745533534128],
            ),
            (
                "z = imag(hilbert([1 3 2 5 4 6]));",
                vec![1.73205080756888, -0.577350269189626, -1.15470053837925, -1.15470053837925, -0.577350269189626, 1.73205080756888],
            ),
        ] {
            assert_close(&reals(src, "z"), &want);
        }
        assert_close(&reals("c = xcorr([1 2 3 4], [1 -1 2]);", "c"), &[0.0, 2.0, 3.0, 5.0, 7.0, -1.0, 4.0]);
        assert_close(&reals("[c, l] = xcorr([1 2 3 4], [1 -1 2]);", "l"), &[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        let err = crate::exec("c = xcorr([1 2 3 4], [1 -1 2], 3, 'unbiased');").err().expect("error");
        assert!(err.to_string().contains("Scale option must be 'none'"), "{err}");
        assert_close(&reals("y = nextpow2([0.3 1 3 1000 -7]);", "y"), &[-1.0, 0.0, 2.0, 10.0, 3.0]);
        assert_close(&reals("y = poly([1 2; 3 4]);", "y"), &[1.0, -5.0, -2.0]);
        assert_close(&reals("y = polyval([1 2 3], [1 2; 3 4]);", "y"), &[6.0, 18.0, 11.0, 27.0]);
        assert_close(&reals("y = ifftshift([1 2 3 4 5; 6 7 8 9 10]);", "y"), &[8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 6.0, 1.0, 7.0, 2.0]);
        assert_close(&reals("q = deconv([1 2 3 4], [1 2]);", "q"), &[1.0, 0.0, 3.0]);
    }

    #[test]
    fn elementwise_helpers() {
        assert_close(&reals("y = nextpow2([1 2 3 1000 0.3 0]);", "y"), &[0.0, 1.0, 2.0, 10.0, -1.0, 0.0]);
        let sinc = reals("y = sinc([0 0.5 1]);", "y");
        assert!((sinc[0] - 1.0).abs() < 1e-12 && (sinc[1] - 2.0 / std::f64::consts::PI).abs() < 1e-12);
        assert!(sinc[2].abs() < 1e-12);
        assert_close(&reals("y = angle(1i);", "y"), &[std::f64::consts::FRAC_PI_2]);
    }

    #[test]
    fn shifts_cumsum_and_diff() {
        assert_close(&reals("y = fftshift([0 1 2 3 4]);", "y"), &[3.0, 4.0, 0.0, 1.0, 2.0]);
        assert_close(&reals("y = fftshift([0 1 2 3]);", "y"), &[2.0, 3.0, 0.0, 1.0]);
        assert_close(&reals("y = ifftshift(fftshift([0 1 2 3 4]));", "y"), &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_close(&reals("y = ifftshift([0 1 2 3 4]);", "y"), &[2.0, 3.0, 4.0, 0.0, 1.0]);
        assert_close(&reals("y = fftshift([1 2; 3 4]);", "y"), &[4.0, 2.0, 3.0, 1.0]);
        assert_close(&reals("y = cumsum([1 2 3]);", "y"), &[1.0, 3.0, 6.0]);
        assert_close(&reals("y = cumsum([1 2; 3 4]);", "y"), &[1.0, 4.0, 2.0, 6.0]);
        assert_close(&reals("y = cumsum([1 2; 3 4], 2);", "y"), &[1.0, 3.0, 3.0, 7.0]);
        assert_close(&reals("y = diff([1 4 9 16]);", "y"), &[3.0, 5.0, 7.0]);
        assert_close(&reals("y = diff([1 4 9 16], 2);", "y"), &[2.0, 2.0]);
        assert!(reals("y = diff(5);", "y").is_empty());
    }

    #[test]
    fn signal_operations_preserve_empty_matrix_shapes() {
        let session = exec(
            "a = zeros(3, 0); c = cumsum(a); c2 = cumsum(a, 2); d = diff(a); d2 = diff(a, 1, 2); h = hilbert(a); u = unwrap(a); t = detrend(a);",
        )
        .unwrap();
        for (name, shape) in [
            ("c", (3, 0)),
            ("c2", (3, 0)),
            ("d", (2, 0)),
            ("d2", (3, 0)),
            ("h", (3, 0)),
            ("u", (3, 0)),
            ("t", (3, 0)),
        ] {
            let matrix = session.get(name).unwrap().clone().into_matrix().unwrap();
            assert_eq!((matrix.rows, matrix.cols), shape, "{name}");
        }
    }

    #[test]
    fn unwrap_removes_two_pi_jumps() {
        let src = "t = 0:0.8:8; y = unwrap(angle(exp(1i * t)));";
        assert_close(&reals(src, "y"), &reals("t = 0:0.8:8;", "t"));
        assert_close(&reals("y = unwrap([0 0.5 6.5 7]);", "y"), &[0.0, 0.5, 6.5 - 2.0 * std::f64::consts::PI, 7.0 - 2.0 * std::f64::consts::PI]);
    }

    #[test]
    fn polynomial_functions() {
        assert_close(&reals("[q, r] = deconv([1 3 5 3], [1 1]);", "q"), &[1.0, 2.0, 3.0]);
        assert_close(&reals("[q, r] = deconv([1 3 5 4], [1 1]);", "r"), &[0.0, 0.0, 0.0, 1.0]);
        assert_close(&reals("q = deconv([1 2], [1 2 3]);", "q"), &[0.0]);
        assert_close(&reals("y = polyval([1 2 3], 2);", "y"), &[11.0]);
        assert_close(&reals("y = polyval([1 0 -1], [1 2 3]);", "y"), &[0.0, 3.0, 8.0]);
        assert_close(&reals("y = poly([1 2]);", "y"), &[1.0, -3.0, 2.0]);
        assert_close(&reals("y = poly([2 0; 0 3]);", "y"), &[1.0, -5.0, 6.0]);
        let mut found = reals("r = roots([1 -3 2]);", "r");
        found.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_close(&found, &[1.0, 2.0]);
        let session = crate::exec("r = roots([1 0 1]);").unwrap();
        let r = session.get("r").unwrap().clone().into_matrix().unwrap();
        assert!(r.data.iter().all(|z| z.re.abs() < 1e-9 && (z.im.abs() - 1.0).abs() < 1e-9));
        assert_close(&reals("r = roots([0 2 -4]);", "r"), &[2.0]);
        assert_close(&reals("r = roots([1 0 0]);", "r"), &[0.0, 0.0]);
        assert!(reals("r = roots([5]);", "r").is_empty());
    }

    #[test]
    fn hilbert_gives_the_analytic_signal() {
        let session = crate::exec("z = hilbert([1 0 -1 0]);").unwrap();
        let z = session.get("z").unwrap().clone().into_matrix().unwrap();
        let want = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        for (got, (re, im)) in z.data.iter().zip(want) {
            assert!((got.re - re).abs() < 1e-9 && (got.im - im).abs() < 1e-9, "{got}");
        }
    }

    #[test]
    fn detrend_removes_constant_and_linear_parts() {
        assert_close(&reals("y = detrend([1 2 3 4]);", "y"), &[0.0, 0.0, 0.0, 0.0]);
        assert_close(&reals("y = detrend([1 1 1], 'constant');", "y"), &[0.0, 0.0, 0.0]);
        assert_close(&reals("y = detrend([1 2 4 8]);", "y"), &[0.7, -0.6, -0.9, 0.8]);
        assert_close(&reals("y = detrend([1 4 9 16 25], 2);", "y"), &[0.0; 5]);
        assert_close(&reals("y = detrend([1 3; 2 5; 3 7]);", "y"), &[0.0; 6]);
    }

    #[test]
    fn xcorr_lags_scaling_and_cross_correlation() {
        assert_close(&reals("c = xcorr([1 2 3]);", "c"), &[3.0, 8.0, 14.0, 8.0, 3.0]);
        assert_close(&reals("[c, l] = xcorr([1 2 3]);", "l"), &[-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert_close(&reals("c = xcorr([1 2 3], 1);", "c"), &[8.0, 14.0, 8.0]);
        assert_close(&reals("c = xcorr([1 2 3], 'coeff');", "c")[2..3], &[1.0]);
        assert_close(&reals("c = xcorr([1 2 3], 'biased');", "c")[2..3], &[14.0 / 3.0]);
        assert_close(&reals("c = xcorr([1 2 3], 'unbiased');", "c")[0..1], &[3.0]);
        assert_close(&reals("c = xcorr([1 2 3], [0 1 0]);", "c"), &[0.0, 1.0, 2.0, 3.0, 0.0]);
        assert_close(&reals("c = xcorr([1; 2; 3]);", "c"), &[3.0, 8.0, 14.0, 8.0, 3.0]);
        assert!(crate::exec("c = xcorr([1 2; 3 4]);").is_err());
        assert!(crate::exec("c = xcorr([1 2], 'nope');").is_err());
    }
}
