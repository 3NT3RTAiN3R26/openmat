use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Matrix, Session, Value};
use rustfft::FftPlanner;

use crate::sigproc::{along, default_dim};
use crate::util::{col, matrix_arg, need, positive, real_scalar, row};

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("fft", fft as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("ifft", ifft),
        ("fft2", fft2),
        ("ifft2", ifft2),
        ("conv", conv),
    ] {
        session.register_builtin(name, func);
    }
}

fn fft(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    transform(args, true)
}

fn ifft(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    transform(args, false)
}

/// `(x, n, dim)`: an empty `n` keeps the length, `dim` defaults to the first non-singleton one.
fn transform(args: &[Value], forward: bool) -> MatResult<Value> {
    need(args, 1)?;
    let m = matrix_arg(args, 0)?;
    let dim = match args.get(2) {
        Some(_) => {
            let d = positive(real_scalar(args, 2)?)?;
            if d != 1 && d != 2 {
                return Err(MatError::new("dimension must be 1 or 2"));
            }
            d
        }
        None => default_dim(&m),
    };
    let length = if dim == 1 { m.rows } else { m.cols };
    let n = match args.get(1) {
        Some(Value::Matrix(size)) if size.numel() == 0 => length,
        Some(_) => positive(real_scalar(args, 1)?)?,
        None => length,
    };
    Ok(Value::Matrix(along(&m, dim, n, |x| {
        fft_buf(x, n, forward)
    })?))
}

fn transform_2d(args: &[Value], forward: bool) -> MatResult<Value> {
    need(args, 1)?;
    let m = matrix_arg(args, 0)?;
    let (rows, cols) = match (args.get(1), args.get(2)) {
        (Some(_), Some(_)) => (
            positive(real_scalar(args, 1)?)?,
            positive(real_scalar(args, 2)?)?,
        ),
        (None, None) => (m.rows, m.cols),
        _ => return Err(MatError::new("fft2 takes either X, or X with both sizes")),
    };
    let columns = along(&m, 1, rows, |x| fft_buf_raw(x, rows, forward))?;
    let both = along(&columns, 2, cols, |x| fft_buf_raw(x, cols, forward))?;
    Ok(Value::Matrix(if forward {
        both
    } else {
        real_if_symmetric(both)
    }))
}

fn fft2(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    transform_2d(args, true)
}

fn ifft2(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    transform_2d(args, false)
}

/// Pads or truncates to `n` samples and transforms. An inverse transform is normalized.
fn fft_buf_raw(samples: &[Complex64], n: usize, forward: bool) -> Vec<Complex64> {
    let mut buf = samples.to_vec();
    buf.resize(n, Complex64::new(0.0, 0.0));
    let mut planner = FftPlanner::<f64>::new();
    let algo = if forward {
        planner.plan_fft_forward(n)
    } else {
        planner.plan_fft_inverse(n)
    };
    algo.process(&mut buf);
    if !forward {
        for v in &mut buf {
            *v /= n as f64;
        }
    }
    buf
}

fn fft_buf(samples: &[Complex64], n: usize, forward: bool) -> Vec<Complex64> {
    let real_input = samples.iter().all(|z| z.im == 0.0);
    let mut out = fft_buf_raw(samples, n, forward);
    if forward && real_input {
        // The spectrum of a real signal is conjugate symmetric, which makes ifft(fft(x)) real.
        for k in 1..n.div_ceil(2) {
            out[n - k] = out[k].conj();
        }
        out[0].im = 0.0;
        if n.is_multiple_of(2) {
            out[n / 2].im = 0.0;
        }
    }
    if !forward {
        let scale = out.iter().map(|z| z.norm()).fold(0.0, f64::max);
        let imag = out.iter().map(|z| z.im.abs()).fold(0.0, f64::max);
        if imag <= 1e-12 * scale {
            for v in &mut out {
                v.im = 0.0;
            }
        }
    }
    out
}

fn real_if_symmetric(mut m: Matrix) -> Matrix {
    let scale = m.data.iter().map(|z| z.norm()).fold(0.0, f64::max);
    let imag = m.data.iter().map(|z| z.im.abs()).fold(0.0, f64::max);
    if imag <= 1e-12 * scale {
        for v in &mut m.data {
            v.im = 0.0;
        }
    }
    m
}

fn conv(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let a_matrix = matrix_arg(args, 0)?;
    let a = a_matrix.reals()?;
    let b = matrix_arg(args, 1)?.reals()?;
    let shape = match args.get(2) {
        None => "full".to_string(),
        Some(Value::Char(shape)) => shape.clone(),
        Some(other) => {
            return Err(MatError::new(format!(
                "conv shape must be 'full', 'same' or 'valid', got {}",
                other.type_name()
            )))
        }
    };
    // The result is a column when the first input is a column vector, otherwise a row.
    let column = a_matrix.cols == 1 && a_matrix.rows > 1;
    let oriented = |values: Vec<f64>| if column { col(values) } else { row(values) };
    if a.is_empty() || b.is_empty() {
        return Ok(oriented(Vec::new()));
    }
    let mut y = vec![0.0; a.len() + b.len() - 1];
    for (i, av) in a.iter().enumerate() {
        for (j, bv) in b.iter().enumerate() {
            y[i + j] += av * bv;
        }
    }
    let y = match shape.as_str() {
        "full" => y,
        "same" => {
            let start = (b.len() - 1).div_ceil(2);
            y[start..start + a.len()].to_vec()
        }
        "valid" if a.len() >= b.len() => y[b.len() - 1..a.len()].to_vec(),
        "valid" => Vec::new(),
        other => {
            return Err(MatError::new(format!(
                "conv shape must be 'full', 'same' or 'valid', got '{other}'"
            )))
        }
    };
    Ok(oriented(y))
}
