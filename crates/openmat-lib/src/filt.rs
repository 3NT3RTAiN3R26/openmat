use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Matrix, Session, Value};

use crate::sigproc::{coefficient_row, default_dim, poly_from_roots, polynomial_roots, xcorr};
use crate::util::{col, matrix_arg, need, positive, real_scalar, row};

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("filter", filter as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("filtic", filtic),
        ("filtfilt", filtfilt),
        ("sosfilt", sosfilt),
        ("sos2tf", sos2tf),
        ("zp2tf", zp2tf),
        ("tf2zp", tf2zp),
        ("freqs", freqs),
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

fn filtfilt(_: &mut Session, args: &[Value]) -> MatResult<Value> {
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
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![coefficient_row(b), coefficient_row(a)]))
    } else {
        Ok(coefficient_row(b))
    }
}

fn complex_column(values: Vec<Complex64>) -> Value {
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

fn grpdelay(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let b = coefficients(args, 0)?;
    let a = if args.len() >= 2 { coefficients(args, 1)? } else { vec![1.0] };
    let n = if args.len() >= 3 { positive(real_scalar(args, 2)?)? } else { 512 };
    if a.is_empty() || b.is_empty() {
        return Err(MatError::new("grpdelay coefficients must not be empty"));
    }
    let reversed: Vec<f64> = a.iter().rev().copied().collect();
    let c = convolve(&b, &reversed);
    let order = (a.len() - 1) as f64;
    let mut delays = Vec::with_capacity(n);
    let mut freqs = Vec::with_capacity(n);
    for i in 0..n {
        let omega = PI * i as f64 / n as f64;
        let mut num = Complex64::new(0.0, 0.0);
        let mut den = Complex64::new(0.0, 0.0);
        for (k, ck) in c.iter().enumerate() {
            let term = Complex64::from_polar(*ck, -omega * k as f64);
            den += term;
            num += term * k as f64;
        }
        delays.push(if den.norm() < 1e-10 { 0.0 } else { (num / den).re - order });
        freqs.push(omega);
    }
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![col(delays), col(freqs)]))
    } else {
        Ok(col(delays))
    }
}

/// A length for the impulse response: the numerator length for FIR filters, otherwise until the
/// slowest pole has decayed to 5e-5. This is a heuristic and not MATLAB's `impzlength`.
fn impulse_length(b: &[f64], a: &[f64]) -> MatResult<usize> {
    let trimmed = a.iter().rposition(|c| *c != 0.0).map_or(0, |last| last + 1);
    if trimmed <= 1 {
        return Ok(b.len());
    }
    let radius = polynomial_roots(&a[..trimmed])?
        .iter()
        .map(|p| p.norm())
        .fold(0.0, f64::max);
    if radius > 0.0 && radius < 1.0 {
        let decay = (5e-5f64.ln() / radius.ln()).ceil() as usize;
        Ok(decay.clamp(b.len(), 100_000))
    } else if radius == 0.0 {
        Ok(b.len().max(trimmed))
    } else {
        Ok(100)
    }
}

fn impz(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let b = coefficients(args, 0)?;
    let a = if args.len() >= 2 { coefficients(args, 1)? } else { vec![1.0] };
    let n = if args.len() >= 3 {
        positive(real_scalar(args, 2)?)?
    } else {
        impulse_length(&b, &a)?
    };
    let (bn, an) = normalized(&b, &a)?;
    let mut impulse = vec![0.0; n];
    if n > 0 {
        impulse[0] = 1.0;
    }
    let (h, _) = lfilter(&bn, &an, &impulse, &[]);
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![col(h), col((0..n).map(|k| k as f64).collect())]))
    } else {
        Ok(col(h))
    }
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
        if second.numel() == 1 && first.numel() > 1 {
            option = second.real_scalar()?;
        } else {
            if second.numel() != first.numel() {
                return Err(MatError::new("cov and corrcoef need inputs of the same length"));
            }
            columns.push(second.reals()?);
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
            data[j * p + i] = if i == j {
                1.0
            } else {
                c.get(i, j).re / (c.get(i, i).re * c.get(j, j).re).sqrt()
            };
        }
    }
    Ok(Value::Matrix(Matrix::from_reals(p, p, data)))
}

/// `xcorr` of the inputs with their means removed.
fn xcov(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let demeaned: Vec<Value> = args
        .iter()
        .enumerate()
        .map(|(i, arg)| match arg {
            Value::Matrix(m) if i < 2 && m.numel() > 1 => {
                let mean = m.data.iter().sum::<Complex64>() / m.numel() as f64;
                Value::Matrix(m.map(|z| z - mean))
            }
            other => other.clone(),
        })
        .collect();
    xcorr(session, &demeaned)
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
}
