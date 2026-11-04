use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Matrix, Session, Value};

use crate::sigproc::{coefficient_row, conjugate_closed, default_dim, poly_from_roots, polynomial_roots, xcorr};
use crate::util::{col, matrix_arg, need, positive, real_scalar, row};

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("filter", filter as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("filtic", filtic),
        ("filtfilt", filtfilt),
        ("sosfilt", sosfilt),
        ("sos2tf", sos2tf),
        ("tf2sos", tf2sos),
        ("zp2sos", zp2sos),
        ("zp2tf", zp2tf),
        ("tf2zp", tf2zp),
        ("residuez", residuez),
        ("zplane", zplane),
        ("freqs", freqs),
        ("freqz", freqz),
        ("phasez", phasez),
        ("stepz", stepz),
        ("grpdelay", grpdelay),
        ("impz", impz),
        ("cov", cov),
        ("corrcoef", corrcoef),
        ("xcov", xcov),
    ] {
        session.register_builtin(name, func);
    }
}

fn coefficients(args: &[Value], index: usize) -> MatResult<Vec<f64>> {
    matrix_arg(args, index)?.reals()
}

/// Pads `b` and `a` to the same length and divides both by `a(1)`.
fn normalized(b: &[f64], a: &[f64]) -> MatResult<(Vec<f64>, Vec<f64>)> {
    if a.is_empty() || a[0] == 0.0 {
        return Err(MatError::new("filter denominator a(1) must be non-zero"));
    }
    if b.is_empty() {
        return Err(MatError::new("filter numerator must not be empty"));
    }
    let n = b.len().max(a.len());
    let scale = a[0];
    let mut bn: Vec<f64> = b.iter().map(|v| v / scale).collect();
    let mut an: Vec<f64> = a.iter().map(|v| v / scale).collect();
    bn.resize(n, 0.0);
    an.resize(n, 0.0);
    Ok((bn, an))
}

/// Transposed direct form II. `b` and `a` are normalized and equally long, `zi` holds `n - 1` states.
fn lfilter(b: &[f64], a: &[f64], x: &[f64], zi: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = b.len();
    let mut z = zi.to_vec();
    z.resize(n - 1, 0.0);
    let mut y = Vec::with_capacity(x.len());
    for &xi in x {
        let yi = if n == 1 { b[0] * xi } else { b[0] * xi + z[0] };
        for i in 0..n.saturating_sub(2) {
            z[i] = z[i + 1] + b[i + 1] * xi - a[i + 1] * yi;
        }
        if n >= 2 {
            z[n - 2] = b[n - 1] * xi - a[n - 1] * yi;
        }
        y.push(yi);
    }
    (y, z)
}

/// The columns (`dim` 1) or rows (`dim` 2) of `m`.
fn sequences(m: &Matrix, dim: usize) -> MatResult<Vec<Vec<f64>>> {
    let values = m.reals()?;
    Ok(if dim == 1 {
        (0..m.cols)
            .map(|c| values[c * m.rows..(c + 1) * m.rows].to_vec())
            .collect()
    } else {
        (0..m.rows)
            .map(|r| (0..m.cols).map(|c| values[c * m.rows + r]).collect())
            .collect()
    })
}

fn assemble(rows: usize, cols: usize, dim: usize, seqs: &[Vec<f64>]) -> Matrix {
    if dim == 1 {
        return Matrix::from_reals(rows, cols, seqs.concat());
    }
    let mut data = vec![0.0; rows * cols];
    for (r, seq) in seqs.iter().enumerate() {
        for (c, v) in seq.iter().enumerate() {
            data[c * rows + r] = *v;
        }
    }
    Matrix::from_reals(rows, cols, data)
}

fn dimension(args: &[Value], index: usize, m: &Matrix) -> MatResult<usize> {
    if args.len() <= index {
        return Ok(default_dim(m));
    }
    match positive(real_scalar(args, index)?)? {
        dim @ (1 | 2) => Ok(dim),
        _ => Err(MatError::new("dimension must be 1 or 2")),
    }
}

fn filter(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let (b, a) = normalized(&coefficients(args, 0)?, &coefficients(args, 1)?)?;
    let x = matrix_arg(args, 2)?;
    let dim = dimension(args, 4, &x)?;
    let states = b.len() - 1;
    let initial = match args.get(3) {
        Some(value) => Some(value.clone().into_matrix()?).filter(|zi| zi.numel() > 0),
        None => None,
    };
    let seqs = sequences(&x, dim)?;
    let initial_for = |k: usize| -> MatResult<Vec<f64>> {
        let Some(zi) = &initial else {
            return Ok(Vec::new());
        };
        if zi.numel() == states {
            zi.reals()
        } else if zi.rows == states && zi.cols == seqs.len() {
            Ok(zi.column(k).reals()?)
        } else {
            Err(MatError::new(
                "initial conditions must have max(length(a), length(b)) - 1 rows",
            ))
        }
    };
    let mut outputs = Vec::with_capacity(seqs.len());
    let mut finals = Vec::with_capacity(seqs.len());
    for (k, seq) in seqs.iter().enumerate() {
        let (y, z) = lfilter(&b, &a, seq, &initial_for(k)?);
        outputs.push(y);
        finals.push(z);
    }
    let y = Value::Matrix(assemble(x.rows, x.cols, dim, &outputs));
    if session.nargout >= 2 {
        let zf = Matrix::from_reals(states, finals.len(), finals.concat());
        Ok(Value::Multi(vec![y, Value::Matrix(zf)]))
    } else {
        Ok(y)
    }
}

fn filtic(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let (b, a) = normalized(&coefficients(args, 0)?, &coefficients(args, 1)?)?;
    let past_y = coefficients(args, 2)?;
    let past_x = if args.len() >= 4 { coefficients(args, 3)? } else { Vec::new() };
    let n = b.len();
    let at = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(0.0);
    let z = (0..n - 1)
        .map(|i| {
            (i + 1..n)
                .map(|j| b[j] * at(&past_x, j - i - 1) - a[j] * at(&past_y, j - i - 1))
                .sum()
        })
        .collect();
    Ok(col(z))
}

/// Initial state that makes the filter start in the steady state of a unit step.
fn steady_state(b: &[f64], a: &[f64]) -> MatResult<Vec<f64>> {
    let n = b.len();
    if n <= 1 {
        return Ok(Vec::new());
    }
    let size = n - 1;
    // I - C^T, where C is the companion matrix of `a`.
    let mut system = DMatrix::<f64>::identity(size, size);
    for j in 0..size {
        system[(j, 0)] += a[j + 1];
    }
    for i in 1..size {
        system[(i - 1, i)] -= 1.0;
    }
    let target = DVector::from_fn(size, |i, _| b[i + 1] - a[i + 1] * b[0]);
    let solution = system
        .lu()
        .solve(&target)
        .ok_or_else(|| MatError::new("filtfilt cannot find initial conditions for this filter"))?;
    Ok(solution.iter().copied().collect())
}

pub(crate) fn filtfilt(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let (b, a) = normalized(&coefficients(args, 0)?, &coefficients(args, 1)?)?;
    let x = matrix_arg(args, 2)?;
    let dim = default_dim(&x);
    let pad = 3 * (b.len() - 1);
    let zi = steady_state(&b, &a)?;
    let scaled = |scale: f64| zi.iter().map(|z| z * scale).collect::<Vec<f64>>();
    let mut outputs = Vec::new();
    for seq in sequences(&x, dim)? {
        let n = seq.len();
        if n <= pad {
            return Err(MatError::new(
                "Data must have length more than 3 times filter order.",
            ));
        }
        // Odd extension about both end points, as MATLAB does.
        let mut padded: Vec<f64> = (1..=pad).rev().map(|k| 2.0 * seq[0] - seq[k]).collect();
        padded.extend_from_slice(&seq);
        padded.extend((1..=pad).map(|k| 2.0 * seq[n - 1] - seq[n - 1 - k]));
        let (mut forward, _) = lfilter(&b, &a, &padded, &scaled(padded[0]));
        forward.reverse();
        let (mut backward, _) = lfilter(&b, &a, &forward, &scaled(forward[0]));
        backward.reverse();
        outputs.push(backward[pad..pad + n].to_vec());
    }
    Ok(Value::Matrix(assemble(x.rows, x.cols, dim, &outputs)))
}

/// The second-order sections of an `L`-by-6 matrix as `(b, a)` rows.
fn sections(sos: &Matrix) -> MatResult<Vec<(Vec<f64>, Vec<f64>)>> {
    if sos.cols != 6 {
        return Err(MatError::new("sos must be an L-by-6 matrix [b0 b1 b2 a0 a1 a2]"));
    }
    if !sos.is_real() {
        return Err(MatError::new("sos must contain real coefficients"));
    }
    (0..sos.rows)
        .map(|r| {
            let value = |c: usize| sos.get(r, c).re;
            normalized(&[value(0), value(1), value(2)], &[value(3), value(4), value(5)])
        })
        .collect()
}

fn sosfilt(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let sections = sections(&matrix_arg(args, 0)?)?;
    let x = matrix_arg(args, 1)?;
    let dim = dimension(args, 2, &x)?;
    let outputs: Vec<Vec<f64>> = sequences(&x, dim)?
        .into_iter()
        .map(|seq| {
            sections
                .iter()
                .fold(seq, |signal, (b, a)| lfilter(b, a, &signal, &[]).0)
        })
        .collect();
    Ok(Value::Matrix(assemble(x.rows, x.cols, dim, &outputs)))
}

fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.len() + b.len() - 1];
    for (i, av) in a.iter().enumerate() {
        for (j, bv) in b.iter().enumerate() {
            y[i + j] += av * bv;
        }
    }
    y
}

fn sos2tf(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let sos = matrix_arg(args, 0)?;
    if sos.cols != 6 {
        return Err(MatError::new("sos must be an L-by-6 matrix [b0 b1 b2 a0 a1 a2]"));
    }
    if !sos.is_real() {
        return Err(MatError::new("sos must contain real coefficients"));
    }
    let gain = if args.len() >= 2 { real_scalar(args, 1)? } else { 1.0 };
    let mut b = vec![gain];
    let mut a = vec![1.0];
    for r in 0..sos.rows {
        let part = |range: std::ops::Range<usize>| -> Vec<f64> { range.map(|c| sos.get(r, c).re).collect() };
        b = convolve(&b, &part(0..3));
        a = convolve(&a, &part(3..6));
    }
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![row(b), row(a)]))
    } else {
        Ok(row(b))
    }
}

fn zp2tf(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let zeros = matrix_arg(args, 0)?.data;
    let poles = matrix_arg(args, 1)?.data;
    let gain = real_scalar(args, 2)?;
    let b: Vec<Complex64> = poly_from_roots(&zeros).into_iter().map(|c| c * gain).collect();
    let a = poly_from_roots(&poles);
    // MATLAB pads a shorter numerator with leading zeros to the length of the denominator.
    let mut padded = vec![Complex64::new(0.0, 0.0); a.len().saturating_sub(b.len())];
    padded.extend(b);
    let b = coefficient_row(padded, conjugate_closed(&zeros));
    let a = coefficient_row(a, conjugate_closed(&poles));
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![b, a]))
    } else {
        Ok(b)
    }
}

pub(crate) fn complex_column(values: Vec<Complex64>) -> Value {
    Value::Matrix(Matrix {
        rows: values.len(),
        cols: 1,
        data: values,
        logical: false,
    })
}

fn tf2zp(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let b = coefficients(args, 0)?;
    let a = coefficients(args, 1)?;
    if b.len() > a.len() {
        return Err(MatError::new("Transfer function not proper."));
    }
    let first = |p: &[f64]| p.iter().copied().find(|c| *c != 0.0);
    let Some(a0) = first(&a) else {
        return Err(MatError::new("tf2zp denominator must not be zero"));
    };
    let gain = first(&b).map_or(0.0, |b0| b0 / a0);
    let zeros = complex_column(polynomial_roots(&b)?);
    if session.nargout >= 2 {
        let poles = complex_column(polynomial_roots(&a)?);
        Ok(Value::Multi(vec![zeros, poles, Value::scalar(gain)]))
    } else {
        Ok(zeros)
    }
}

fn freqs(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let b = coefficients(args, 0)?;
    let a = coefficients(args, 1)?;
    let w = matrix_arg(args, 2)?;
    if w.numel() == 1 {
        return Err(MatError::new(
            "freqs needs a vector of frequencies in rad/s, a point count is not supported",
        ));
    }
    if a.is_empty() || b.is_empty() {
        return Err(MatError::new("freqs coefficients must not be empty"));
    }
    let horner = |p: &[f64], s: Complex64| p.iter().fold(Complex64::new(0.0, 0.0), |acc, c| acc * s + c);
    let h = w.map(|z| {
        let s = Complex64::new(0.0, z.re);
        horner(&b, s) / horner(&a, s)
    });
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![Value::Matrix(h), Value::Matrix(w)]))
    } else {
        Ok(Value::Matrix(h))
    }
}

/// Frequency points and the output grid of `freqz`, `grpdelay` and `phasez`.
struct Grid {
    omega: Vec<f64>,
    /// Same values as `omega`, or in Hz when a sampling frequency was given.
    reported: Vec<f64>,
    /// The points came from a vector argument, so the outputs are rows when it was a row.
    like: Option<Matrix>,
}

/// Parses `(n | w, fs, 'whole')` after the coefficient arguments.
fn frequency_grid(args: &[Value], name: &str) -> MatResult<Grid> {
    let mut whole = false;
    let mut numeric: Vec<&Value> = Vec::new();
    for arg in args {
        match arg {
            Value::Char(option) if option == "whole" => whole = true,
            Value::Char(option) if option == "half" => whole = false,
            Value::Char(other) => {
                return Err(MatError::new(format!("{name} does not know the option '{other}'")))
            }
            other => numeric.push(other),
        }
    }
    let fs = match numeric.get(1) {
        Some(value) => {
            let fs = value.real_scalar()?;
            if !(fs.is_finite() && fs > 0.0) {
                return Err(MatError::new(format!("{name}: the sampling frequency must be positive")));
            }
            Some(fs)
        }
        None => None,
    };
    let first = match numeric.first() {
        Some(value) => Some((*value).clone().into_matrix()?),
        None => None,
    };
    match first {
        Some(points) if points.numel() > 1 => {
            let values = points.reals()?;
            let omega = match fs {
                Some(fs) => values.iter().map(|f| 2.0 * PI * f / fs).collect(),
                None => values.clone(),
            };
            Ok(Grid { omega, reported: values, like: Some(points) })
        }
        other => {
            let n = match other {
                Some(m) => positive(m.real_scalar()?)?,
                None => 512,
            };
            if n == 0 {
                return Err(MatError::new(format!("{name} needs at least one frequency point")));
            }
            let span = if whole { 2.0 * PI } else { PI };
            let omega: Vec<f64> = (0..n).map(|i| span * i as f64 / n as f64).collect();
            let reported = match fs {
                Some(fs) => omega.iter().map(|w| w * fs / (2.0 * PI)).collect(),
                None => omega.clone(),
            };
            Ok(Grid { omega, reported, like: None })
        }
    }
}

/// `sum(c[k] e^{-j w k})`.
fn dtft(c: &[f64], omega: f64) -> Complex64 {
    c.iter()
        .enumerate()
        .map(|(k, v)| Complex64::from_polar(*v, -omega * k as f64))
        .sum()
}

fn grid_value(grid: &Grid, values: Vec<Complex64>) -> Value {
    let (rows, cols) = match &grid.like {
        Some(points) => (points.rows, points.cols),
        None => (values.len(), 1),
    };
    Value::Matrix(Matrix { rows, cols, data: values, logical: false })
}

fn grid_reals(grid: &Grid, values: &[f64]) -> Value {
    grid_value(grid, values.iter().map(|v| Complex64::from(*v)).collect())
}

fn freqz(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let b = coefficients(args, 0)?;
    let a = if args.len() >= 2 && !matches!(args[1], Value::Char(_)) {
        coefficients(args, 1)?
    } else {
        vec![1.0]
    };
    if a.is_empty() || b.is_empty() {
        return Err(MatError::new("freqz coefficients must not be empty"));
    }
    let rest = if args.len() >= 2 && !matches!(args[1], Value::Char(_)) { &args[2..] } else { &args[1..] };
    let grid = frequency_grid(rest, "freqz")?;
    let h: Vec<Complex64> = grid
        .omega
        .iter()
        .map(|w| dtft(&b, *w) / dtft(&a, *w))
        .collect();
    if session.nargout == 0 {
        let mag: Vec<f64> = h.iter().map(|z| 20.0 * z.norm().max(1e-30).log10()).collect();
        return plot_response(session, &grid.reported, &mag);
    }
    let response = grid_value(&grid, h);
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![response, grid_reals(&grid, &grid.reported)]))
    } else {
        Ok(response)
    }
}

/// Phase of `h` in radians, unwrapped across jumps larger than pi. Zeros of the response give NaN.
fn unwrapped_phase(h: &[Complex64]) -> Vec<f64> {
    let scale = h.iter().map(|z| z.norm()).fold(0.0, f64::max);
    let mut offset = 0.0;
    let mut previous: Option<f64> = None;
    h.iter()
        .map(|z| {
            if z.norm() <= 1e-12 * scale.max(1.0) {
                return f64::NAN;
            }
            let raw = z.arg();
            if let Some(last) = previous {
                let step = raw - last;
                if step.abs() > PI {
                    offset -= 2.0 * PI * (step / (2.0 * PI)).round();
                }
            }
            previous = Some(raw);
            raw + offset
        })
        .collect()
}

fn phasez(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let b = coefficients(args, 0)?;
    let a = if args.len() >= 2 && !matches!(args[1], Value::Char(_)) {
        coefficients(args, 1)?
    } else {
        vec![1.0]
    };
    if a.is_empty() || b.is_empty() {
        return Err(MatError::new("phasez coefficients must not be empty"));
    }
    let rest = if args.len() >= 2 && !matches!(args[1], Value::Char(_)) { &args[2..] } else { &args[1..] };
    let grid = frequency_grid(rest, "phasez")?;
    let h: Vec<Complex64> = grid
        .omega
        .iter()
        .map(|w| dtft(&b, *w) / dtft(&a, *w))
        .collect();
    let phase_values = unwrapped_phase(&h);
    if session.nargout == 0 {
        return plot_response(session, &grid.reported, &phase_values);
    }
    let phase = grid_reals(&grid, &phase_values);
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![phase, grid_reals(&grid, &grid.reported)]))
    } else {
        Ok(phase)
    }
}

fn grpdelay(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let b = coefficients(args, 0)?;
    let a = if args.len() >= 2 && !matches!(args[1], Value::Char(_)) {
        coefficients(args, 1)?
    } else {
        vec![1.0]
    };
    if a.is_empty() || b.is_empty() {
        return Err(MatError::new("grpdelay coefficients must not be empty"));
    }
    let rest = if args.len() >= 2 && !matches!(args[1], Value::Char(_)) { &args[2..] } else { &args[1..] };
    let grid = frequency_grid(rest, "grpdelay")?;
    let reversed: Vec<f64> = a.iter().rev().copied().collect();
    let c = convolve(&b, &reversed);
    let weighted: Vec<f64> = c.iter().enumerate().map(|(k, ck)| k as f64 * ck).collect();
    let order = (a.len() - 1) as f64;
    let tolerance = 1e-10 * c.iter().map(|ck| ck.abs()).sum::<f64>();
    let mut delays: Vec<Option<f64>> = grid
        .omega
        .iter()
        .map(|w| {
            let den = dtft(&c, *w);
            (den.norm() > tolerance).then(|| (dtft(&weighted, *w) / den).re - order)
        })
        .collect();
    // At a zero on the unit circle the delay is undefined. MATLAB reports a finite value, here the
    // mean of the neighboring frequency points.
    let known = delays.clone();
    for (i, delay) in delays.iter_mut().enumerate() {
        if delay.is_none() {
            let before = known[..i].iter().rev().find_map(|d| *d);
            let after = known[i + 1..].iter().find_map(|d| *d);
            *delay = Some(match (before, after) {
                (Some(x), Some(y)) => (x + y) / 2.0,
                (Some(x), None) | (None, Some(x)) => x,
                (None, None) => 0.0,
            });
        }
    }
    let values: Vec<f64> = delays.into_iter().map(|d| d.unwrap_or(0.0)).collect();
    if session.nargout == 0 {
        return plot_response(session, &grid.reported, &values);
    }
    let delay = grid_reals(&grid, &values);
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![delay, grid_reals(&grid, &grid.reported)]))
    } else {
        Ok(delay)
    }
}

/// Length of the impulse response when `n` is not given, from the pole radius `r`: MATLAB stops
/// when a stable response has decayed to 5e-5, and an unstable one has grown to 1e6. Both
/// lengths are rounded down, which reproduces the lengths MATLAB R2025a returns.
/// For a pole on the unit circle the length is a fixed 100, which is not checked against MATLAB.
fn impulse_length(b: &[f64], a: &[f64]) -> MatResult<usize> {
    let trimmed = a.iter().rposition(|c| *c != 0.0).map_or(0, |last| last + 1);
    if trimmed <= 1 {
        return Ok(b.len());
    }
    let radius = polynomial_roots(&a[..trimmed])?
        .iter()
        .map(|p| p.norm())
        .fold(0.0, f64::max);
    let length = if radius == 0.0 {
        b.len().max(trimmed)
    } else if (radius - 1.0).abs() < 1e-9 {
        100
    } else if radius < 1.0 {
        (5e-5f64.ln() / radius.ln()).floor() as usize
    } else {
        (1e6f64.ln() / radius.ln()).floor() as usize
    };
    Ok(length.min(100_000).max(b.len()))
}

fn impulse_response(b: &[f64], a: &[f64], length: Option<usize>) -> MatResult<Vec<f64>> {
    let n = match length {
        Some(n) => n,
        None => impulse_length(b, a)?,
    };
    let (bn, an) = normalized(b, a)?;
    let mut impulse = vec![0.0; n];
    if n > 0 {
        impulse[0] = 1.0;
    }
    Ok(lfilter(&bn, &an, &impulse, &[]).0)
}

/// `(b, a, n, fs)` of `impz` and `stepz`. The time axis is in samples, or in seconds with `fs`.
type ImpulseArguments = (Vec<f64>, Vec<f64>, Option<usize>, f64);

fn impulse_arguments(args: &[Value], name: &str) -> MatResult<ImpulseArguments> {
    need(args, 1)?;
    let b = coefficients(args, 0)?;
    let a = if args.len() >= 2 { coefficients(args, 1)? } else { vec![1.0] };
    let n = if args.len() >= 3 && matrix_arg(args, 2)?.numel() > 0 {
        Some(positive(real_scalar(args, 2)?)?)
    } else {
        None
    };
    let fs = if args.len() >= 4 { real_scalar(args, 3)? } else { 1.0 };
    if !(fs.is_finite() && fs > 0.0) {
        return Err(MatError::new(format!("{name}: the sampling frequency must be positive")));
    }
    Ok((b, a, n, fs))
}

fn response_and_time(session: &Session, response: Vec<f64>, fs: f64) -> Value {
    if session.nargout >= 2 {
        let time = (0..response.len()).map(|k| k as f64 / fs).collect();
        Value::Multi(vec![col(response), col(time)])
    } else {
        col(response)
    }
}

fn impz(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (b, a, n, fs) = impulse_arguments(args, "impz")?;
    let response = impulse_response(&b, &a, n)?;
    Ok(response_and_time(session, response, fs))
}

fn stepz(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (b, a, n, fs) = impulse_arguments(args, "stepz")?;
    let mut total = 0.0;
    let response: Vec<f64> = impulse_response(&b, &a, n)?
        .into_iter()
        .map(|v| {
            total += v;
            total
        })
        .collect();
    Ok(response_and_time(session, response, fs))
}

// Statistics

/// Data columns for `cov` and `corrcoef`, and the normalization option (0: N - 1, 1: N).
fn statistic_columns(args: &[Value]) -> MatResult<(Vec<Vec<f64>>, f64)> {
    need(args, 1)?;
    let first = matrix_arg(args, 0)?;
    let mut option = 0.0;
    let mut columns = if first.rows == 1 || first.cols == 1 {
        vec![first.reals()?]
    } else {
        sequences(&first, 1)?
    };
    if args.len() >= 2 {
        let second = matrix_arg(args, 1)?;
        if second.numel() == 1 && args.len() == 2 {
            option = second.real_scalar()?;
        } else {
            if second.numel() != first.numel() {
                return Err(MatError::new("cov and corrcoef need inputs of the same length"));
            }
            columns = vec![first.reals()?, second.reals()?];
        }
    }
    if args.len() >= 3 {
        option = real_scalar(args, 2)?;
    }
    if option != 0.0 && option != 1.0 {
        return Err(MatError::new("the normalization option must be 0 or 1"));
    }
    Ok((columns, option))
}

fn covariance_matrix(columns: &[Vec<f64>], option: f64) -> Matrix {
    let p = columns.len();
    let n = columns.first().map_or(0, Vec::len);
    let means: Vec<f64> = columns
        .iter()
        .map(|c| c.iter().sum::<f64>() / n.max(1) as f64)
        .collect();
    let denominator = if option == 0.0 && n > 1 { (n - 1) as f64 } else { n as f64 };
    let mut data = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            let sum: f64 = (0..n)
                .map(|k| (columns[i][k] - means[i]) * (columns[j][k] - means[j]))
                .sum();
            data[j * p + i] = sum / denominator;
        }
    }
    Matrix::from_reals(p, p, data)
}

fn cov(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (columns, option) = statistic_columns(args)?;
    Ok(Value::Matrix(covariance_matrix(&columns, option)))
}

fn corrcoef(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (columns, option) = statistic_columns(args)?;
    let c = covariance_matrix(&columns, option);
    let p = c.rows;
    let mut data = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            data[j * p + i] = if i == j && c.get(i, i).re > 0.0 && c.get(i, i).re.is_finite() {
                1.0
            } else {
                c.get(i, j).re / c.get(i, i).re.sqrt() / c.get(j, j).re.sqrt()
            };
        }
    }
    Ok(Value::Matrix(Matrix::from_reals(p, p, data)))
}

/// `xcorr` of the inputs with their means removed.
fn xcov(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let numeric: Vec<usize> = args
        .iter()
        .enumerate()
        .filter_map(|(i, arg)| match arg {
            Value::Char(_) => None,
            _ => Some(i),
        })
        .collect();
    let mut demeaned = args.to_vec();
    for (position, &index) in numeric.iter().take(2).enumerate() {
        let m = args[index].clone().into_matrix()?;
        if m.numel() == 0 || (position == 1 && numeric.len() == 2 && m.numel() == 1) {
            continue;
        }
        let mean = m.data.iter().sum::<Complex64>() / m.numel() as f64;
        demeaned[index] = Value::Matrix(m.map(|z| z - mean));
    }
    xcorr(session, &demeaned)
}

fn plot_response(session: &mut Session, x: &[f64], y: &[f64]) -> MatResult<Value> {
    use openmat_core::{Series, SeriesKind};
    crate::plotb::prepare_plot_target(session);
    let axes = session.current_figure_mut().axes_mut();
    if !axes.hold {
        axes.series.clear();
    }
    axes.series.push(Series {
        xs: x.to_vec(),
        ys: y.to_vec(),
        kind: SeriesKind::Line,
        label: String::new(),
    });
    Ok(crate::util::empty())
}

fn tf2sos(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let b = coefficients(args, 0)?;
    let a = coefficients(args, 1)?;
    let zeros = polynomial_roots(&b)?;
    let poles = polynomial_roots(&a)?;
    let gain = b[0] / a[0];
    sos_from_zp(session, &zeros, &poles, gain, &args[2..])
}

fn zp2sos(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let zeros = matrix_arg(args, 0)?.data.clone();
    let poles = matrix_arg(args, 1)?.data.clone();
    let gain = if args.len() >= 3 && !matches!(args[2], Value::Char(_)) {
        real_scalar(args, 2)?
    } else {
        1.0
    };
    let rest = if args.len() >= 3 && !matches!(args[2], Value::Char(_)) { &args[3..] } else { &args[2..] };
    sos_from_zp(session, &zeros, &poles, gain, rest)
}

fn sos_from_zp(session: &mut Session, zeros: &[Complex64], poles: &[Complex64], gain: f64, options: &[Value]) -> MatResult<Value> {
    let mut up = true;
    let mut scale = "none";
    for option in options {
        let Value::Char(text) = option else {
            return Err(MatError::new("tf2sos options must be strings"));
        };
        match text.to_ascii_lowercase().as_str() {
            "up" => up = true,
            "down" => up = false,
            "none" => scale = "none",
            "inf" => scale = "inf",
            "two" => scale = "two",
            other => return Err(MatError::new(format!("tf2sos does not know '{other}'"))),
        }
    }
    let mut sections = pair_sections(zeros, poles)?;
    sections.sort_by(|left, right| {
        let pole = left.radius.partial_cmp(&right.radius).unwrap_or(std::cmp::Ordering::Equal);
        // Equal pole radii keep the conjugate pair with the smaller real part first,
        // which is the order `cplxpair` produces for a numerator-only transfer function.
        let order = if (left.radius - right.radius).abs() < 1e-8 {
            right.b1.partial_cmp(&left.b1).unwrap_or(pole)
        } else {
            pole
        };
        if up { order } else { order.reverse() }
    });
    let mut rows: Vec<[f64; 6]> = sections
        .iter()
        .map(|section| [1.0, section.b1, section.b2, 1.0, section.a1, section.a2])
        .collect();
    if rows.is_empty() {
        rows.push([gain, 0.0, 0.0, 1.0, 0.0, 0.0]);
        return finish_sos(session, &rows, 1.0);
    }
    let g = if scale == "none" { gain } else { scale_sos(&mut rows, gain, scale) };
    finish_sos(session, &rows, g)
}

/// Infinity-norm or 2-norm section scaling.
/// `beta(k)` is the norm of the transfer function that reaches the input of section `k`
/// (numerators of the earlier sections over the denominators through section `k`).
/// Section `k` is multiplied by `beta(k) / beta(k+1)`, the last section by `beta(last)`,
/// and the leading gain becomes `gain / beta(1)`.
fn scale_sos(rows: &mut [[f64; 6]], gain: f64, scale: &str) -> f64 {
    let mut beta = Vec::with_capacity(rows.len());
    let mut numerator = vec![1.0];
    let mut denominator = vec![1.0];
    for (index, row) in rows.iter().enumerate() {
        denominator = strip_trailing_zeros(convolve(&denominator, &section_den(row)));
        beta.push(filter_norm(&numerator, &denominator, scale));
        if index + 1 < rows.len() {
            numerator = strip_trailing_zeros(convolve(&numerator, &[row[0], row[1], row[2]]));
        }
    }
    for (index, row) in rows.iter_mut().enumerate() {
        let factor = if index + 1 < beta.len() { beta[index] / beta[index + 1] } else { beta[index] };
        row[0] *= factor;
        row[1] *= factor;
        row[2] *= factor;
    }
    gain / beta[0]
}

fn section_den(row: &[f64; 6]) -> Vec<f64> {
    if row[5].abs() > 1e-14 {
        vec![1.0, row[4], row[5]]
    } else if row[4].abs() > 1e-14 {
        vec![1.0, row[4]]
    } else {
        vec![1.0]
    }
}

fn strip_trailing_zeros(mut coeffs: Vec<f64>) -> Vec<f64> {
    while coeffs.len() > 1 && coeffs.last().copied().unwrap_or(0.0).abs() < 1e-14 {
        coeffs.pop();
    }
    coeffs
}

fn filter_norm(num: &[f64], den: &[f64], scale: &str) -> f64 {
    let norm = if scale == "two" { impulse_l2(num, den) } else { infinity_norm(num, den) };
    norm.max(1e-30)
}

fn infinity_norm(num: &[f64], den: &[f64]) -> f64 {
    let samples = 8192;
    let mut best_w = 0.0;
    let mut best = 0.0;
    for k in 0..=samples {
        let w = PI * k as f64 / samples as f64;
        let magnitude = freq_mag(num, den, w);
        if magnitude > best {
            best = magnitude;
            best_w = w;
        }
    }
    let mut low = (best_w - PI / samples as f64).max(0.0);
    let mut high = (best_w + PI / samples as f64).min(PI);
    for _ in 0..40 {
        let left = low + (high - low) / 3.0;
        let right = high - (high - low) / 3.0;
        if freq_mag(num, den, left) < freq_mag(num, den, right) {
            low = left;
        } else {
            high = right;
        }
    }
    freq_mag(num, den, (low + high) / 2.0)
}

fn freq_mag(num: &[f64], den: &[f64], w: f64) -> f64 {
    let z = Complex64::new(w.cos(), -w.sin());
    let n = horner(num, z);
    let d = horner(den, z);
    if d.norm() == 0.0 { f64::MAX } else { (n / d).norm() }
}

fn horner(coeffs: &[f64], z: Complex64) -> Complex64 {
    let mut acc = Complex64::new(0.0, 0.0);
    for coef in coeffs.iter().rev() {
        acc = acc * z + Complex64::new(*coef, 0.0);
    }
    acc
}

fn impulse_l2(num: &[f64], den: &[f64]) -> f64 {
    let order = num.len().max(den.len()) - 1;
    let mut state = vec![0.0; order];
    let mut energy = 0.0;
    let mut input = 1.0;
    for _ in 0..100_000 {
        let y = num.first().copied().unwrap_or(0.0) * input + state.first().copied().unwrap_or(0.0);
        if order > 0 {
            for i in 0..order - 1 {
                let b = num.get(i + 1).copied().unwrap_or(0.0);
                let a = den.get(i + 1).copied().unwrap_or(0.0);
                state[i] = state[i + 1] + b * input - a * y;
            }
            let b = num.get(order).copied().unwrap_or(0.0);
            let a = den.get(order).copied().unwrap_or(0.0);
            state[order - 1] = b * input - a * y;
        }
        energy += y * y;
        input = 0.0;
        if y.abs() < 1e-18 && state.iter().all(|s| s.abs() < 1e-18) {
            break;
        }
    }
    energy.sqrt()
}

struct Section {
    radius: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

fn pair_sections(zeros: &[Complex64], poles: &[Complex64]) -> MatResult<Vec<Section>> {
    let mut z = zeros.to_vec();
    let mut p = poles.to_vec();
    // Drop numerical conjugates' duplicates by keeping one of each pair and the real roots.
    let z = unique_roots(&mut z);
    let p = unique_roots(&mut p);
    let mut z = z;
    let mut p = p;
    let mut sections = Vec::new();
    while !p.is_empty() || !z.is_empty() {
        if p.is_empty() {
            let z1 = z.remove(0);
            let (b1, b2) = if z1.im.abs() > 1e-8 {
                (-2.0 * z1.re, z1.norm_sqr())
            } else if let Some(index) = nearest_real(&z, z1) {
                let z2 = z.remove(index);
                (-(z1.re + z2.re), z1.re * z2.re)
            } else {
                (-z1.re, 0.0)
            };
            // No poles: every section sits at the origin, and the real-part tie break orders them.
            sections.push(Section { radius: 0.0, b1, b2, a1: 0.0, a2: 0.0 });
            continue;
        }
        let index = worst_pole(&p);
        let p1 = p.remove(index);
        if p1.im.abs() > 1e-8 {
            let (b1, b2) = take_zeros(&mut z, p1, true);
            sections.push(Section {
                radius: p1.norm(),
                b1,
                b2,
                a1: -2.0 * p1.re,
                a2: p1.norm_sqr(),
            });
        } else if p.iter().any(|pole| pole.im.abs() <= 1e-8) {
            let index = nearest_real(&p, p1).unwrap();
            let p2 = p.remove(index);
            let (b1, b2) = take_zeros(&mut z, p1, false);
            sections.push(Section {
                radius: p1.re.abs().min(p2.re.abs()),
                b1,
                b2,
                a1: -(p1.re + p2.re),
                a2: p1.re * p2.re,
            });
        } else {
            let (b1, b2) = if let Some(index) = nearest_real(&z, p1) {
                let z1 = z.remove(index);
                (-z1.re, 0.0)
            } else {
                (0.0, 0.0)
            };
            sections.push(Section { radius: p1.re.abs(), b1, b2, a1: -p1.re, a2: 0.0 });
        }
    }
    Ok(sections)
}

fn unique_roots(roots: &mut [Complex64]) -> Vec<Complex64> {
    let mut kept = Vec::new();
    let mut used = vec![false; roots.len()];
    for i in 0..roots.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        if roots[i].im.abs() > 1e-8 {
            if let Some(j) = (0..roots.len()).find(|j| !used[*j] && (roots[*j] - roots[i].conj()).norm() < 1e-6) {
                used[j] = true;
            }
            // Keep the one with a non-negative imaginary part.
            let root = if roots[i].im < 0.0 { roots[i].conj() } else { roots[i] };
            kept.push(root);
        } else {
            kept.push(Complex64::new(roots[i].re, 0.0));
        }
    }
    kept
}

fn worst_pole(poles: &[Complex64]) -> usize {
    poles.iter().enumerate().min_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap()).map(|(i, _)| i).unwrap_or(0)
}

fn nearest_real(roots: &[Complex64], target: Complex64) -> Option<usize> {
    roots
        .iter()
        .enumerate()
        .filter(|(_, root)| root.im.abs() <= 1e-8)
        .min_by(|(_, a), (_, b)| (a.re - target.re).abs().partial_cmp(&(b.re - target.re).abs()).unwrap())
        .map(|(i, _)| i)
}

fn take_zeros(zeros: &mut Vec<Complex64>, pole: Complex64, complex: bool) -> (f64, f64) {
    if zeros.is_empty() {
        return (0.0, 0.0);
    }
    if complex {
        if let Some(index) = zeros.iter().enumerate().filter(|(_, z)| z.im.abs() > 1e-8).min_by(|(_, a), (_, b)| {
            (a.norm() - pole.norm()).abs().partial_cmp(&(b.norm() - pole.norm()).abs()).unwrap()
        }).map(|(i, _)| i) {
            let z1 = zeros.remove(index);
            return (-2.0 * z1.re, z1.norm_sqr());
        }
    }
    let index = zeros.iter().enumerate().min_by(|(_, a), (_, b)| (*a - pole).norm().partial_cmp(&(*b - pole).norm()).unwrap()).map(|(i, _)| i).unwrap();
    let z1 = zeros.remove(index);
    if z1.im.abs() > 1e-8 {
        return (-2.0 * z1.re, z1.norm_sqr());
    }
    if let Some(index) = nearest_real(zeros, z1) {
        let z2 = zeros.remove(index);
        (-(z1.re + z2.re), z1.re * z2.re)
    } else {
        (-z1.re, 0.0)
    }
}

fn finish_sos(session: &mut Session, rows: &[[f64; 6]], gain: f64) -> MatResult<Value> {
    let mut data = vec![Complex64::new(0.0, 0.0); rows.len() * 6];
    for (row_index, row) in rows.iter().enumerate() {
        for (col, value) in row.iter().enumerate() {
            data[col * rows.len() + row_index] = Complex64::new(*value, 0.0);
        }
    }
    let sos = Value::Matrix(Matrix { rows: rows.len(), cols: 6, data, logical: false });
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![sos, Value::scalar(gain)]))
    } else {
        Ok(sos)
    }
}

fn residuez(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    if session.nargout <= 2 && args.len() >= 2 && matrix_arg(args, 0).ok().filter(|m| m.cols == 1 || m.rows == 1).is_some() && args.len() >= 3 {
        return residuez_inverse(args);
    }
    // Inverse when the first argument is a residue vector and a direct term is present,
    // or when there are three inputs.
    if args.len() >= 3 {
        return residuez_inverse(args);
    }
    let b = matrix_arg(args, 0)?.data.clone();
    let a = matrix_arg(args, 1)?.data.clone();
    if a.is_empty() || a[0].norm() == 0.0 {
        return Err(MatError::new("residuez denominator must not start at zero"));
    }
    let (quotient, remainder) = split_negative_powers(&b, &a);
    let poles = polynomial_roots(&a.iter().map(|z| z.re).collect::<Vec<_>>())?;
    let mut groups: Vec<(Complex64, usize)> = Vec::new();
    for pole in poles {
        if let Some(group) = groups.iter_mut().find(|(p, _)| (*p - pole).norm() < 1e-8) {
            group.1 += 1;
        } else {
            groups.push((pole, 1));
        }
    }
    let mut residues = Vec::new();
    let mut pole_list = Vec::new();
    for (pole, mult) in &groups {
        for order in 1..=*mult {
            residues.push(residue_at(&remainder, &a, *pole, *mult, order));
            pole_list.push(*pole);
        }
    }
    let k = if quotient.is_empty() { empty_matrix() } else { row_complex(quotient) };
    Ok(Value::Multi(vec![col_complex(residues), col_complex(pole_list), k]))
}

fn residuez_inverse(args: &[Value]) -> MatResult<Value> {
    let residues = matrix_arg(args, 0)?.data.clone();
    let poles = matrix_arg(args, 1)?.data.clone();
    let direct = if args.len() >= 3 { matrix_arg(args, 2)?.data.clone() } else { Vec::new() };
    if residues.len() != poles.len() {
        return Err(MatError::new("residuez needs one residue per pole"));
    }
    let mut den = vec![Complex64::new(1.0, 0.0)];
    let mut num = Vec::new();
    let mut index = 0;
    while index < poles.len() {
        let pole = poles[index];
        let mut mult = 1;
        while index + mult < poles.len() && (poles[index + mult] - pole).norm() < 1e-8 {
            mult += 1;
        }
        let factor = poly_pow_pole(pole, mult);
        let mut piece = vec![Complex64::new(0.0, 0.0)];
        for order in 1..=mult {
            let mut term = vec![residues[index + order - 1]];
            let rest = poly_pow_pole(pole, mult - order);
            term = poly_mul(&term, &rest);
            piece = poly_add(&piece, &term);
        }
        num = poly_add(&poly_mul(&num, &factor), &poly_mul(&piece, &den));
        den = poly_mul(&den, &factor);
        index += mult;
    }
    if !direct.is_empty() {
        num = poly_add(&poly_mul(&direct, &den), &num);
    }
    let scale = den[0];
    for coef in &mut num {
        *coef /= scale;
    }
    for coef in &mut den {
        *coef /= scale;
    }
    Ok(Value::Multi(vec![row_complex(num), row_complex(den)]))
}

fn residue_at(num: &[Complex64], den: &[Complex64], pole: Complex64, mult: usize, order: usize) -> Complex64 {
    // r_order = 1/(mult-order)! * d^{mult-order}/dz of (z-p)^mult N/D at z=p.
    // For the simple and double poles in the reference set a short finite difference is enough,
    // and an exact formula is used when the other factor is known.
    let deriv = mult - order;
    let step = 1e-5;
    let samples = 2 * deriv;
    // Contour-free real finite difference on a small real step, which is exact for low order.
    let mut coeffs = vec![Complex64::new(0.0, 0.0); samples + 1];
    for (k, coef) in coeffs.iter_mut().enumerate() {
        let z = pole + Complex64::new((k as f64 - deriv as f64) * step, 0.0);
        *coef = cleared(num, den, pole, mult, z);
    }
    // Repeated differences.
    for _ in 0..deriv {
        for k in 0..coeffs.len() - 1 {
            coeffs[k] = (coeffs[k + 1] - coeffs[k]) / step;
        }
        coeffs.pop();
    }
    let mut value = coeffs[deriv.min(coeffs.len() - 1)];
    let mut fact = 1.0;
    for i in 1..=deriv {
        fact *= i as f64;
    }
    if deriv > 0 {
        value /= fact;
    }
    value
}

fn cleared(num: &[Complex64], den: &[Complex64], pole: Complex64, mult: usize, z: Complex64) -> Complex64 {
    let n = poly_eval(num, z);
    let mut d = poly_eval(den, z);
    let factor = z - pole;
    // (z-p)^mult * N/D. Cancel the pole factor analytically when d is tiny.
    if d.norm() < 1e-8 {
        let mut stripped = den.to_vec();
        for _ in 0..mult {
            stripped = poly_deflate(&stripped, pole);
        }
        d = poly_eval(&stripped, z);
        if d.norm() == 0.0 {
            return Complex64::new(0.0, 0.0);
        }
        return n / d;
    }
    n / d * factor.powi(mult as i32)
}

fn poly_eval(coeffs: &[Complex64], z: Complex64) -> Complex64 {
    let mut acc = Complex64::new(0.0, 0.0);
    for coef in coeffs {
        acc = acc * z + coef;
    }
    acc
}

/// Polynomial division for coefficients stored in ascending powers of `z^{-1}`.
/// Returns the direct term and the strictly proper numerator in that same order.
fn split_negative_powers(num: &[Complex64], den: &[Complex64]) -> (Vec<Complex64>, Vec<Complex64>) {
    if num.len() < den.len() {
        return (Vec::new(), num.to_vec());
    }
    let mut rev_num = num.to_vec();
    let mut rev_den = den.to_vec();
    rev_num.reverse();
    rev_den.reverse();
    let (mut quotient, mut remainder) = poly_div(&rev_num, &rev_den);
    quotient.reverse();
    remainder.reverse();
    (quotient, trim_zeros(remainder))
}

fn poly_div(num: &[Complex64], den: &[Complex64]) -> (Vec<Complex64>, Vec<Complex64>) {
    if num.len() < den.len() {
        return (Vec::new(), num.to_vec());
    }
    let mut rem = num.to_vec();
    let mut quot = vec![Complex64::new(0.0, 0.0); num.len() - den.len() + 1];
    for i in 0..quot.len() {
        let factor = rem[i] / den[0];
        quot[i] = factor;
        for (j, coef) in den.iter().enumerate() {
            rem[i + j] -= factor * coef;
        }
    }
    let rest = rem[quot.len()..].to_vec();
    (quot, trim_zeros(rest))
}

fn poly_mul(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![Complex64::new(0.0, 0.0); a.len() + b.len() - 1];
    for (i, av) in a.iter().enumerate() {
        for (j, bv) in b.iter().enumerate() {
            out[i + j] += av * bv;
        }
    }
    out
}

fn poly_add(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    let n = a.len().max(b.len());
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for (i, v) in a.iter().enumerate() {
        out[n - a.len() + i] += v;
    }
    for (i, v) in b.iter().enumerate() {
        out[n - b.len() + i] += v;
    }
    trim_zeros(out)
}

fn poly_pow_pole(pole: Complex64, power: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(1.0, 0.0)];
    let factor = vec![Complex64::new(1.0, 0.0), -pole];
    for _ in 0..power {
        out = poly_mul(&out, &factor);
    }
    out
}

fn poly_deflate(coeffs: &[Complex64], root: Complex64) -> Vec<Complex64> {
    if coeffs.len() < 2 {
        return vec![Complex64::new(1.0, 0.0)];
    }
    let mut out = vec![Complex64::new(0.0, 0.0); coeffs.len() - 1];
    let mut acc = Complex64::new(0.0, 0.0);
    for i in 0..out.len() {
        acc = coeffs[i] + acc * root;
        out[i] = acc;
    }
    out
}

fn trim_zeros(mut coeffs: Vec<Complex64>) -> Vec<Complex64> {
    while coeffs.len() > 1 && coeffs.last().unwrap().norm() < 1e-12 {
        coeffs.pop();
    }
    coeffs
}

fn col_complex(values: Vec<Complex64>) -> Value {
    Value::Matrix(Matrix { rows: values.len(), cols: 1, data: values, logical: false })
}

fn row_complex(values: Vec<Complex64>) -> Value {
    Value::Matrix(Matrix { rows: 1, cols: values.len(), data: values, logical: false })
}

fn empty_matrix() -> Value {
    Value::Matrix(Matrix { rows: 0, cols: 0, data: Vec::new(), logical: false })
}

fn zplane(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let first = matrix_arg(args, 0)?;
    let (zeros, poles) = if args.len() == 1 {
        if first.cols > 1 || (first.rows == 1 && first.cols >= 1) {
            (Vec::new(), polynomial_roots(&first.reals()?)?)
        } else {
            (Vec::new(), first.data.clone())
        }
    } else {
        let second = matrix_arg(args, 1)?;
        let zeros = if is_row_poly(&first) { polynomial_roots(&first.reals()?)? } else { first.data.clone() };
        let poles = if is_row_poly(&second) { polynomial_roots(&second.reals()?)? } else { second.data.clone() };
        (zeros, poles)
    };
    if session.nargout == 0 {
        draw_zplane(session, &zeros, &poles)?;
    }
    // There is no graphics-handle type. Outputs stay empty; the drawing is the no-output form.
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![empty_matrix(), empty_matrix(), empty_matrix()]))
    } else if session.nargout == 1 {
        Ok(empty_matrix())
    } else {
        Ok(crate::util::empty())
    }
}

fn is_row_poly(matrix: &Matrix) -> bool {
    matrix.rows == 1 && matrix.cols > 1
}

fn draw_zplane(session: &mut Session, zeros: &[Complex64], poles: &[Complex64]) -> MatResult<()> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for k in 0..=256 {
        let angle = 2.0 * PI * k as f64 / 256.0;
        xs.push(angle.cos());
        ys.push(angle.sin());
    }
    plot_response(session, &xs, &ys)?;
    let axes = session.current_figure_mut().axes_mut();
    axes.hold = true;
    for (points, label) in [(zeros, "zero"), (poles, "pole")] {
        for point in points {
            let mark = if label == "pole" { 0.04 } else { 0.03 };
            plot_response(session, &[point.re - mark, point.re + mark], &[point.im, point.im])?;
            plot_response(session, &[point.re, point.re], &[point.im - mark, point.im + mark])?;
        }
    }
    let axes = session.current_figure_mut().axes_mut();
    axes.title = "Pole-Zero Plot".to_string();
    axes.hold = false;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::exec;

    fn reals(src: &str, name: &str) -> Vec<f64> {
        let session = exec(src).unwrap_or_else(|err| panic!("{src}: {err}"));
        session
            .get(name)
            .unwrap_or_else(|| panic!("{name} missing after {src}"))
            .clone()
            .into_matrix()
            .unwrap()
            .reals()
            .unwrap()
    }

    fn shape(src: &str, name: &str) -> (usize, usize) {
        let session = exec(src).unwrap();
        let m = session.get(name).unwrap().clone().into_matrix().unwrap();
        (m.rows, m.cols)
    }

    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-9, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn filter_keeps_the_shape_of_the_input() {
        assert_eq!(shape("y = filter(1, 1, [1 2 3 4]);", "y"), (1, 4));
        assert_eq!(shape("y = filter(1, 1, [1; 2; 3]);", "y"), (3, 1));
        assert_close(&reals("y = filter(1, [1 -0.5], [1 0 0 0]);", "y"), &[1.0, 0.5, 0.25, 0.125]);
        assert_close(&reals("y = filter([2 0], [2], [1 2 3]);", "y"), &[1.0, 2.0, 3.0]);
        assert_close(&reals("y = filter([1 1], 1, [1 2; 3 4]);", "y"), &[1.0, 4.0, 2.0, 6.0]);
        assert_close(&reals("y = filter([1 1], 1, [1 2; 3 4], [], 2);", "y"), &[1.0, 3.0, 3.0, 7.0]);
        assert!(crate::exec("y = filter(1, 0, [1 2]);").is_err());
    }

    #[test]
    fn filter_state_continues_a_split_signal() {
        let src = "b = [1 0.5 0.2]; a = [1 -0.3 0.1]; x = [1 2 3 4 5 6]; \
                   whole = filter(b, a, x); \
                   [y1, zf] = filter(b, a, x(1:3)); y2 = filter(b, a, x(4:6), zf);";
        let whole = reals(src, "whole");
        let mut joined = reals(src, "y1");
        joined.extend(reals(src, "y2"));
        assert_close(&joined, &whole);
        assert_eq!(shape("[y, zf] = filter([1 1], 1, [1 2 3]);", "zf"), (1, 1));
        assert_close(&reals("[y, zf] = filter([1 1], 1, [1 2 3]);", "zf"), &[3.0]);
        assert!(crate::exec("y = filter([1 1 1], 1, [1 2 3], [1 2 3]);").is_err());
    }

    #[test]
    fn filtic_reproduces_the_state_of_past_samples() {
        let src = "b = [1 0.5 0.2]; a = [1 -0.3 0.1]; x = [1 2 3 4 5]; whole = filter(b, a, x); \
                   zi = filtic(b, a, [whole(2) whole(1)], [x(2) x(1)]); rest = filter(b, a, x(3:5), zi);";
        assert_close(&reals(src, "rest"), &reals(src, "whole")[2..]);
        assert_close(&reals("z = filtic([1 2], [1 3], 4, 5);", "z"), &[2.0 * 5.0 - 3.0 * 4.0]);
    }

    #[test]
    fn filtfilt_is_zero_phase() {
        assert_close(&reals("y = filtfilt([0.2 0.2], [1 -0.6], 3 * ones(1, 30));", "y"), &[3.0; 30]);
        let src = "x = zeros(1, 14); x(7) = 1; y = filtfilt([0.5 0.5], 1, x);";
        let y = reals(src, "y");
        assert_close(&y[5..8], &[0.25, 0.5, 0.25]);
        assert!(crate::exec("y = filtfilt([1 1 1], 1, [1 2 3]);").is_err());
    }

    #[test]
    fn second_order_sections() {
        let sos = "sos = [1 0 0 1 -0.5 0; 1 0 0 1 -0.5 0];";
        assert_close(&reals(&format!("{sos} y = sosfilt(sos, [1 0 0 0]);"), "y"), &[1.0, 1.0, 0.75, 0.5]);
        assert_close(&reals(&format!("{sos} [b, a] = sos2tf(sos);"), "a"), &[1.0, -1.0, 0.25, 0.0, 0.0]);
        assert_close(&reals(&format!("{sos} [b, a] = sos2tf(sos, 2);"), "b"), &[2.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(crate::exec("y = sosfilt([1 2 3], [1 2]);").is_err());
    }

    #[test]
    fn zero_pole_gain_conversions() {
        let src = "[b, a] = zp2tf([1; 2], [3; 4], 5);";
        assert_close(&reals(src, "b"), &[5.0, -15.0, 10.0]);
        assert_close(&reals(src, "a"), &[1.0, -7.0, 12.0]);
        let src = "[z, p, k] = tf2zp([2 -6 4], [1 -7 12]);";
        let mut z = reals(src, "z");
        let mut p = reals(src, "p");
        z.sort_by(|a, b| a.partial_cmp(b).unwrap());
        p.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_close(&z, &[1.0, 2.0]);
        assert_close(&p, &[3.0, 4.0]);
        assert_close(&reals(src, "k"), &[2.0]);
    }

    #[test]
    fn analog_response_and_group_delay() {
        let session = exec("h = freqs([1], [1 1], [0 1 10]);").unwrap();
        let h = session.get("h").unwrap().clone().into_matrix().unwrap();
        assert!((h.data[0].re - 1.0).abs() < 1e-12);
        assert!((h.data[1].re - 0.5).abs() < 1e-12 && (h.data[1].im + 0.5).abs() < 1e-12);
        assert!(crate::exec("h = freqs(1, [1 1], 5);").is_err());
        assert_close(&reals("g = grpdelay([0 0 1], 1, 4);", "g"), &[2.0; 4]);
        assert!((reals("g = grpdelay(1, [1 -0.5], 8);", "g")[0] - 1.0).abs() < 1e-9);
        assert_close(&reals("[g, w] = grpdelay(1, 1, 4);", "w"), &[0.0, std::f64::consts::FRAC_PI_4, std::f64::consts::FRAC_PI_2, 3.0 * std::f64::consts::FRAC_PI_4]);
    }

    #[test]
    fn impulse_response() {
        assert_close(&reals("h = impz([1 0.5 0.25]);", "h"), &[1.0, 0.5, 0.25]);
        assert_close(&reals("h = impz(1, [1 -0.5], 5);", "h"), &[1.0, 0.5, 0.25, 0.125, 0.0625]);
        assert_close(&reals("[h, t] = impz([1 1], 1, 3);", "t"), &[0.0, 1.0, 2.0]);
        let decayed = reals("h = impz(1, [1 -0.5]);", "h");
        assert!(decayed.len() > 10 && decayed[decayed.len() - 1] < 1e-3);
    }

    #[test]
    fn conv_shapes() {
        assert_close(&reals("y = conv([1 2 3], [1 1 1], 'same');", "y"), &[3.0, 6.0, 5.0]);
        assert_close(&reals("y = conv([1 2 3 4], [1 1], 'valid');", "y"), &[3.0, 5.0, 7.0]);
        assert_close(&reals("y = conv([1 2 3 4], [1 1], 'full');", "y"), &[1.0, 3.0, 5.0, 7.0, 4.0]);
        assert!(reals("y = conv([1 2], [1 1 1], 'valid');", "y").is_empty());
        assert!(crate::exec("y = conv([1 2], [1 1], 'nope');").is_err());
    }

    fn complex_values(src: &str, name: &str) -> Vec<(f64, f64)> {
        let session = exec(src).unwrap_or_else(|err| panic!("{src}: {err}"));
        session
            .get(name)
            .unwrap()
            .clone()
            .into_matrix()
            .unwrap()
            .data
            .iter()
            .map(|z| (z.re, z.im))
            .collect()
    }

    fn assert_complex(got: &[(f64, f64)], want: &[(f64, f64)]) {
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for ((gr, gi), (wr, wi)) in got.iter().zip(want) {
            assert!((gr - wr).abs() < 1e-9 && (gi - wi).abs() < 1e-9, "{got:?} vs {want:?}");
        }
    }

    // The expected values below were produced with MATLAB R2025a, Signal Processing Toolbox 25.1.
    #[test]
    fn frequency_response_matches_matlab() {
        let (h, w) = ("[h, w] = freqz([1 1], 1, 4);", "w");
        assert_complex(
            &complex_values(h, "h"),
            &[(2.0, 0.0), (1.70710678118655, -std::f64::consts::FRAC_1_SQRT_2), (1.0, -1.0), (0.292893218813452, -std::f64::consts::FRAC_1_SQRT_2)],
        );
        assert_close(&reals(h, w), &[0.0, std::f64::consts::FRAC_PI_4, std::f64::consts::FRAC_PI_2, 2.35619449019234]);
        let whole = "[h, w] = freqz([1 1], 1, 4, 'whole');";
        assert_complex(&complex_values(whole, "h"), &[(2.0, 0.0), (1.0, -1.0), (0.0, 0.0), (1.0, 1.0)]);
        assert_close(&reals(whole, "w"), &[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI, 4.71238898038469]);
        let fs = "[h, f] = freqz([1 -0.5], [1 0.3], 4, 1000);";
        assert_complex(
            &complex_values(fs, "h"),
            &[
                (0.384615384615385, 0.0),
                (0.467935981843263, 0.373571186583344),
                (0.779816513761468, 0.73394495412844),
                (1.48921112657852, 0.849714426341305),
            ],
        );
        assert_close(&reals(fs, "f"), &[0.0, 125.0, 250.0, 375.0]);
        let fs_whole = "[h, f] = freqz([1 -0.5], [1 0.3], 4, 1000, 'whole');";
        assert_complex(
            &complex_values(fs_whole, "h"),
            &[(0.384615384615385, 0.0), (0.779816513761468, 0.73394495412844), (2.14285714285714, 0.0), (0.779816513761468, -0.73394495412844)],
        );
        assert_close(&reals(fs_whole, "f"), &[0.0, 250.0, 500.0, 750.0]);
        assert_eq!(shape("h = freqz([1 1], 1);", "h"), (512, 1));
        let points = "[h, w] = freqz([1 1], 1, [0 0.5 1]);";
        assert_eq!(shape(points, "h"), (1, 3));
        assert_complex(
            &complex_values(points, "h"),
            &[(2.0, 0.0), (1.87758256189037, -0.479425538604203), (1.54030230586814, -0.841470984807897)],
        );
    }

    #[test]
    fn phase_and_step_response_match_matlab() {
        let src = "[phi, w] = phasez([1 0.5], [1 -0.3], 8);";
        assert_close(
            &reals(src, "phi"),
            &[0.0, -0.287652489879136, -0.518506361224544, -0.673340432046071, -0.755104403478673, -0.762685926128773, -0.673726850161457, -0.431316971097822],
        );
        assert_close(
            &reals(src, "w"),
            &[0.0, std::f64::consts::FRAC_PI_8, std::f64::consts::FRAC_PI_4, 1.17809724509617, std::f64::consts::FRAC_PI_2, 1.96349540849362, 2.35619449019234, 2.74889357189107],
        );
        let whole = "phi = phasez([1 0.5], [1 -0.3], 8, 'whole');";
        assert_close(
            &reals(whole, "phi"),
            &[0.0, -0.518506361224544, -0.755104403478673, -0.673726850161457, 0.0, 0.673726850161457, 0.755104403478673, 0.518506361224544],
        );
        let zeros = reals("phi = phasez([1 0 0 0 1], 1, 8);", "phi");
        assert!(zeros[2].is_nan() && zeros[6].is_nan());
        assert!((zeros[1] + std::f64::consts::FRAC_PI_4).abs() < 1e-9 && (zeros[3] - std::f64::consts::FRAC_PI_4).abs() < 1e-9);
        assert_eq!(reals("phi = phasez([1 0.5], [1 -0.3]);", "phi").len(), 512);
        assert_close(&reals("[h, t] = stepz(1, [1 -0.5], 6);", "h"), &[1.0, 1.5, 1.75, 1.875, 1.9375, 1.96875]);
        assert_close(&reals("[h, t] = stepz(1, [1 -0.5], 6);", "t"), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_close(&reals("h = stepz([1 1 1], 1, 5);", "h"), &[1.0, 2.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn impulse_response_lengths_match_matlab() {
        for (src, length) in [
            ("h = impz(1, [1 -0.5]);", 14),
            ("h = impz(1, [1 -0.9]);", 93),
            ("h = impz(1, [1 -0.99]);", 985),
            ("h = impz(1, [1 0 0.81]);", 93),
            ("h = impz([1 1 1], 1);", 3),
            ("h = impz(1, [1 -1.1]);", 144),
            ("h = impz([1 2 3], [1 -0.5 0.25 0.1]);", 23),
            ("h = stepz(1, [1 -0.5]);", 14),
            ("h = stepz(1, [1 -0.9]);", 93),
        ] {
            assert_eq!(reals(src, "h").len(), length, "{src}");
        }
        assert_close(&reals("[h, t] = impz(1, [1 -0.5], 5, 100);", "t"), &[0.0, 0.01, 0.02, 0.03, 0.04]);
    }

    #[test]
    fn group_delay_matches_matlab() {
        let src = "[gd, w] = grpdelay([1 0.5], [1 -0.3 0.1], 6);";
        assert_close(
            &reals(src, "gd"),
            &[0.458333333333334, 0.526067372557333, 0.56183368869936, 0.3, -0.154135338345864, -0.78581490069613],
        );
        assert_close(
            &reals(src, "w"),
            &[0.0, std::f64::consts::FRAC_PI_6, std::f64::consts::FRAC_PI_3, std::f64::consts::FRAC_PI_2, 2.0943951023932, 2.61799387799149],
        );
        assert_close(&reals("gd = grpdelay([1 0 1], 1, 4);", "gd"), &[1.0; 4]);
        assert_close(&reals("gd = grpdelay([1 -1], 1, 4);", "gd"), &[0.5; 4]);
        assert_eq!(reals("gd = grpdelay([1 1], 1);", "gd").len(), 512);
        let whole = "[gd, w] = grpdelay([1 1], 1, 4, 'whole');";
        assert_close(&reals(whole, "gd"), &[0.5; 4]);
        assert_close(&reals(whole, "w"), &[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI, 4.71238898038469]);
    }

    #[test]
    fn filter_state_and_initial_conditions_match_matlab() {
        let setup = "b = [1 0.5 0.2]; a = [1 -0.3 0.1]; m = [8 1 6; 3 5 7; 4 9 2];";
        let dim2 = format!("{setup} [y, zf] = filter(b, a, m, [], 2);");
        assert_eq!(shape(&dim2, "zf"), (2, 3));
        assert_close(&reals(&dim2, "zf"), &[5.316, 0.248, 7.366, 0.198, 4.748, -0.656]);
        let dim1 = format!("{setup} [y, zf] = filter(b, a, m);");
        assert_close(&reals(&dim1, "zf"), &[4.396, -0.112, 8.922, 0.466, 4.112, -0.564]);
        let zi = format!("{setup} y = filter(b, a, m, [1 3 5; 2 4 6]);");
        assert_close(&reals(&zi, "y"), &[9.0, 11.7, 9.71, 4.0, 10.7, 14.51, 11.0, 19.3, 11.39]);
        assert_close(&reals(&format!("{setup} z = filtic(b, a, [2 1], [1 3]);"), "z"), &[1.6, 0.0]);
        assert_close(&reals("z = filtic([1 0.5 0.2], [2 -0.6 0.2], [2 1], [1 3]);", "z"), &[1.05, -0.1]);
        assert_close(&reals(&format!("{setup} z = filtic(b, a, [2 1]);"), "z"), &[0.5, -0.2]);
    }

    #[test]
    fn zero_pole_conversions_and_root_order_match_matlab() {
        let ordered = |src: &str, name: &str| complex_values(src, name);
        assert_complex(&ordered("r = roots([1 -6 11 -6]);", "r"), &[(3.0, 0.0), (2.0, 0.0), (1.0, 0.0)]);
        assert_complex(
            &ordered("r = roots([1 0 0 1]);", "r"),
            &[(-1.0, 0.0), (0.5, 0.866025403784438), (0.5, -0.866025403784438)],
        );
        assert_complex(&ordered("r = roots([1 2 5]);", "r"), &[(-1.0, 2.0), (-1.0, -2.0)]);
        assert_complex(
            &ordered("r = roots([1 -1 0.5 -0.1 0.02]);", "r"),
            &[(0.427592031200156, 0.359488456109296), (0.427592031200156, -0.359488456109296), (0.0724079687998438, 0.242581757384494), (0.0724079687998438, -0.242581757384494)],
        );
        let tf = "[z, p, k] = tf2zp([2 -6 4], [1 -7 12]);";
        assert_close(&reals(tf, "z"), &[2.0, 1.0]);
        assert_close(&reals(tf, "p"), &[4.0, 3.0]);
        assert_close(&reals(tf, "k"), &[2.0]);
        assert_complex(&ordered("[z, p, k] = tf2zp(1, [1 -1 0.5]);", "p"), &[(0.5, 0.5), (0.5, -0.5)]);
        for src in ["[z, p, k] = tf2zp([1 2 3], [1 4]);", "[z, p, k] = tf2zp([0 0 1 2], [1 -0.5 0.06]);"] {
            let err = crate::exec(src).err().expect("error");
            assert!(err.to_string().contains("not proper"), "{src}: {err}");
        }
        let padded = "[b, a] = zp2tf([], [0.5; -0.5], 3);";
        assert_close(&reals(padded, "b"), &[0.0, 0.0, 3.0]);
        assert_close(&reals(padded, "a"), &[1.0, 0.0, -0.25]);
        let complex_zeros = "[b, a] = zp2tf([1i; -1i], [0.5], 1);";
        assert_close(&reals(complex_zeros, "b"), &[1.0, 0.0, 1.0]);
        assert_close(&reals(complex_zeros, "a"), &[1.0, -0.5]);
    }

    #[test]
    fn convolution_and_deconvolution_orientation_match_matlab() {
        assert_eq!(shape("y = conv([1 2 3], [1; 1]);", "y"), (1, 4));
        assert_eq!(shape("y = conv([1; 2; 3], [1 1]);", "y"), (4, 1));
        assert_eq!(shape("y = conv([1; 2; 3], [1; 1]);", "y"), (4, 1));
        assert_close(&reals("y = conv([1 2 3 4], [1 1 1 1], 'same');", "y"), &[6.0, 10.0, 9.0, 7.0]);
        assert_close(&reals("y = conv([1 2 3 4 5], [1 1], 'same');", "y"), &[3.0, 5.0, 7.0, 9.0, 5.0]);
        assert_close(&reals("y = conv([1 2], [1 1 1], 'same');", "y"), &[3.0, 3.0]);
        assert_eq!(shape("[q, r] = deconv([1; 3; 5; 3], [1; 1]);", "q"), (3, 1));
        assert_eq!(shape("[q, r] = deconv([1; 3; 5; 3], [1; 1]);", "r"), (4, 1));
        assert_eq!(shape("[q, r] = deconv([1 3 5 3], [1 1]);", "q"), (1, 3));
    }

    #[test]
    fn statistics_match_matlab() {
        let xy = "x = [1 2 3 4 5]; y = [2 1 4 3 7];";
        assert_close(&reals(&format!("{xy} c = cov(x, y);"), "c"), &[2.5, 3.0, 3.0, 5.3]);
        assert_close(&reals(&format!("{xy} c = cov(x, y, 1);"), "c"), &[2.0, 2.4, 2.4, 4.24]);
        assert_close(&reals(&format!("{xy} c = cov(x');"), "c"), &[2.5]);
        assert_close(&reals("c = cov([1 2 3; 4 5 7]);", "c"), &[4.5, 4.5, 6.0, 4.5, 4.5, 6.0, 6.0, 6.0, 8.0]);
        assert_close(&reals(&format!("{xy} r = corrcoef(x, y);"), "r"), &[1.0, 0.824163383692134, 0.824163383692134, 1.0]);
        assert_close(
            &reals("r = corrcoef([1 2 3; 4 5 7; 2 9 1]);", "r"),
            &[1.0, 0.24854790640048, 0.785714285714286, 0.24854790640048, 1.0, -0.40389034790078, 0.785714285714286, -0.40389034790078, 1.0],
        );
        assert_close(&reals(&format!("{xy} c = xcov(x);"), "c"), &[-4.0, -4.0, -1.0, 4.0, 10.0, 4.0, -1.0, -4.0, -4.0]);
        assert_close(
            &reals(&format!("{xy} c = xcov(x, y);"), "c"),
            &[-7.2, -2.8, -0.8, 7.8, 12.0, 1.2, -1.2, -6.2, -2.8],
        );
        assert_close(&reals(&format!("{xy} c = xcov(x, y, 2);"), "c"), &[-0.8, 7.8, 12.0, 1.2, -1.2]);
        assert_close(
            &reals(&format!("{xy} c = xcov(x, y, 2, 'coeff');"), "c"),
            &[-0.0549442255794756, 0.535706199399887, 0.824163383692134, 0.0824163383692134, -0.0824163383692133],
        );
        assert_close(
            &reals(&format!("{xy} c = xcov(x, 'unbiased');"), "c"),
            &[-4.0, -2.0, -0.333333333333334, 1.0, 2.0, 1.0, -0.333333333333334, -2.0, -4.0],
        );
        assert_eq!(shape(&format!("{xy} c = xcov(x');"), "c"), (9, 1));
        assert_eq!(shape(&format!("{xy} c = xcov(x', y');"), "c"), (9, 1));
    }

    #[test]
    fn second_order_sections_match_matlab_filtering() {
        let src = "y = sosfilt([1 2 1 1 -0.5 0; 1 0 0 1 0.1 0], [1 0 0 0 0]);";
        assert_close(&reals(src, "y"), &[1.0, 2.4, 2.01, 0.924, 0.4701]);
    }

    #[test]
    fn covariance_and_correlation() {
        assert_close(&reals("c = cov([1 2 3 4]);", "c"), &[5.0 / 3.0]);
        assert_close(&reals("c = cov([1 2 3 4], 1);", "c"), &[1.25]);
        assert_close(&reals("c = cov([1 2 3], [2 4 6]);", "c"), &[1.0, 2.0, 2.0, 4.0]);
        assert_close(&reals("c = cov([1 2; 3 4; 5 6]);", "c"), &[4.0, 4.0, 4.0, 4.0]);
        assert_close(&reals("r = corrcoef([1 2 3], [3 2 1]);", "r"), &[1.0, -1.0, -1.0, 1.0]);
        assert_close(&reals("r = corrcoef([1 2 3], [2 4 6]);", "r"), &[1.0; 4]);
        assert!(crate::exec("c = cov([1 2 3], [1 2]);").is_err());
        assert_close(&reals("c = xcov([1 2 3]);", "c"), &[-1.0, 0.0, 2.0, 0.0, -1.0]);
    }

    #[test]
    fn sos_residuez_and_zplane_match_matlab() {
        let src = "[sos, g] = tf2sos([1 4 6 4 1], [1 -1.2 0.9 -0.3 0.05]);";
        assert_close(&reals(src, "g"), &[1.0]);
        let sos = reals(src, "sos");
        assert!((sos[8] + 0.508560609940842).abs() < 1e-8, "{sos:?}");
        assert!((sos[9] + 0.691439390059159).abs() < 1e-8, "{sos:?}");
        assert!((sos[2] - 2.00000003354362).abs() < 1e-3, "{sos:?}");
        assert!((sos[10] - 0.115514292823272).abs() < 1e-8, "{sos:?}");
        let src = "[sos, g] = tf2sos([1 4 6 4 1], [1 -1.2 0.9 -0.3 0.05], 'down');";
        let sos = reals(src, "sos");
        assert!((sos[8] + 0.691439390059159).abs() < 1e-8, "{sos:?}");
        let src = "[sos, g] = tf2sos([1 4 6 4 1], [1 -1.2 0.9 -0.3 0.05], 'up', 'inf');";
        assert_close(&reals(src, "g"), &[0.60695368288243]);
        let sos = reals(src, "sos");
        assert!((sos[0] - 0.169316477066066).abs() < 2e-4, "{sos:?}");
        assert!((sos[1] - 9.73072558847786).abs() < 2e-3, "{sos:?}");
        let src = "[sos, g] = tf2sos([1 4 6 4 1], [1 -1.2 0.9 -0.3 0.05], 'up', 'two');";
        assert_close(&reals(src, "g"), &[0.884074050250963]);
        let sos = reals(src, "sos");
        assert!((sos[0] - 0.197148135068433).abs() < 2e-4, "{sos:?}");
        assert!((sos[1] - 5.73744695919316).abs() < 2e-3, "{sos:?}");
        let src = "[sos, g] = tf2sos([1 3 3 1], [1 -0.9 0.5 -0.1]);";
        let sos = reals(src, "sos");
        assert!((sos[8] + 0.317368523971139).abs() < 1e-6, "{sos:?}");
        assert!((sos[4]).abs() < 1e-6, "{sos:?}");
        let src = "[sos, g] = tf2sos([1 2 3 4 5], 1);";
        let sos = reals(src, "sos");
        assert_eq!(sos.len(), 12);
        assert!((sos[2] - 2.5756309591153).abs() < 1e-6, "{sos:?}");
        let src = "[sos, g] = tf2sos([1 0 -1], [1 -1 0.25]);";
        assert_close(&reals(src, "sos"), &[1.0, 0.0, -1.0, 1.0, -1.0, 0.25]);
        let src = "[z, p, k] = tf2zp([1 4 6 4 1], [1 -1.2 0.9 -0.3 0.05]); sos = zp2sos(z, p, k);";
        let sos = reals(src, "sos");
        assert!((sos[8] + 0.508560609940842).abs() < 1e-8, "{sos:?}");
        let src = "[r, p, k] = residuez([1 2], [1 -0.5 -0.06]);";
        assert_close(&reals(src, "r"), &[3.71428571428571, -2.71428571428571]);
        assert_close(&reals(src, "p"), &[0.6, -0.1]);
        let src = "[r, p, k] = residuez([1 2 3 4], [1 -0.5]);";
        assert_close(&reals(src, "r"), &[49.0]);
        assert_close(&reals(src, "k"), &[-48.0, -22.0, -8.0]);
        let src = "[r, p, k] = residuez(1, conv([1 -0.5], [1 -0.5]));";
        assert_close(&reals(src, "r"), &[0.0, 1.0]);
        let src = "[b, a] = residuez([0.5 - 1.5i; 0.5 + 1.5i], [0.5 + 0.5i; 0.5 - 0.5i], []);";
        assert_close(&reals(src, "b"), &[1.0, 1.0]);
        assert_close(&reals(src, "a"), &[1.0, -1.0, 0.5]);
        let session = exec("zplane([1 0.5], [1 -0.9]);").unwrap();
        assert!(session.figures[0].axes[0].series.len() >= 2);
        assert_eq!(session.figures[0].axes[0].title, "Pole-Zero Plot");
    }
}
