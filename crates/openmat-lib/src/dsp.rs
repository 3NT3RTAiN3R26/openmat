use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Matrix, Session, Value};
use rustfft::FftPlanner;

use crate::util::{col, matrix_arg, need, positive, real_scalar, row};

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("fft", fft as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("ifft", ifft),
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

fn transform(args: &[Value], forward: bool) -> MatResult<Value> {
    let m = matrix_arg(args, 0)?;
    let n_default = if m.rows == 1 { m.cols } else { m.rows };
    let n = if args.len() >= 2 {
        positive(real_scalar(args, 1)?)?
    } else {
        n_default
    };
    if m.cols == 1 || m.rows == 1 {
        let mut samples = m.data.clone();
        samples.resize(n, Complex64::new(0.0, 0.0));
        let out = fft_buf(&samples, forward);
        return if m.rows == 1 {
            Ok(Value::Matrix(Matrix {
                rows: 1,
                cols: n,
                data: out,
                logical: false,
            }))
        } else {
            Ok(Value::Matrix(Matrix {
                rows: n,
                cols: 1,
                data: out,
                logical: false,
            }))
        };
    }
    let mut data = vec![Complex64::new(0.0, 0.0); n * m.cols];
    for c in 0..m.cols {
        let mut samples = vec![Complex64::new(0.0, 0.0); n];
        for r in 0..m.rows.min(n) {
            samples[r] = m.get(r, c);
        }
        let out = fft_buf(&samples, forward);
        for r in 0..n {
            data[c * n + r] = out[r];
        }
    }
    Ok(Value::Matrix(Matrix {
        rows: n,
        cols: m.cols,
        data,
        logical: false,
    }))
}

fn fft_buf(samples: &[Complex64], forward: bool) -> Vec<Complex64> {
    let mut planner = FftPlanner::<f64>::new();
    let algo = if forward {
        planner.plan_fft_forward(samples.len())
    } else {
        planner.plan_fft_inverse(samples.len())
    };
    let mut buf = samples.to_vec();
    algo.process(&mut buf);
    if !forward {
        let scale = samples.len() as f64;
        for v in &mut buf {
            *v /= scale;
            if v.im.abs() < 1e-8 {
                v.im = 0.0;
            }
            if v.re.abs() < 1e-8 {
                v.re = 0.0;
            }
        }
    }
    buf
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
