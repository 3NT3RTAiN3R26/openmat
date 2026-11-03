use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Matrix, Session, Value};
use rustfft::FftPlanner;

use crate::util::{col, matrix_arg, need, positive, real_scalar, row};

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("fft", fft as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("ifft", ifft),
        ("conv", conv),
        ("freqz", freqz),
        ("fir1", fir1),
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
    let a = matrix_arg(args, 0)?.reals()?;
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
    if a.is_empty() || b.is_empty() {
        return Ok(row(Vec::new()));
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
    Ok(row(y))
}

fn freqz(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let b = matrix_arg(args, 0)?.reals()?;
    let a = matrix_arg(args, 1)?.reals()?;
    let n = if args.len() >= 3 {
        positive(real_scalar(args, 2)?)?.max(1)
    } else {
        64
    };
    let (h, w) = frequency_response(&b, &a, n);
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![
            Value::Matrix(Matrix {
                rows: n,
                cols: 1,
                data: h,
                logical: false,
            }),
            col(w),
        ]))
    } else {
        Ok(Value::Matrix(Matrix {
            rows: n,
            cols: 1,
            data: h,
            logical: false,
        }))
    }
}

pub fn frequency_response(b: &[f64], a: &[f64], n: usize) -> (Vec<Complex64>, Vec<f64>) {
    let mut h = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let omega = if n == 1 {
            0.0
        } else {
            std::f64::consts::PI * i as f64 / (n - 1) as f64
        };
        let z = Complex64::from_polar(1.0, -omega);
        let num = polyval_complex(b, z);
        let den = polyval_complex(a, z);
        h.push(num / den);
        w.push(omega);
    }
    (h, w)
}

fn polyval_complex(coeff_high_first: &[f64], z: Complex64) -> Complex64 {
    let mut acc = Complex64::new(0.0, 0.0);
    for c in coeff_high_first {
        acc = acc * z + c;
    }
    acc
}

fn fir1(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let order = positive(real_scalar(args, 0)?)?;
    let wn = real_scalar(args, 1)?;
    if !(0.0..1.0).contains(&wn) {
        return Err(MatError::new("fir1 wn must be in (0, 1)"));
    }
    Ok(row(fir_window(order, wn)))
}

pub fn fir_window(order: usize, wn: f64) -> Vec<f64> {
    let wc = wn * std::f64::consts::PI;
    let mid = order as f64 / 2.0;
    let mut h = vec![0.0; order + 1];
    for i in 0..=order {
        let x = i as f64 - mid;
        let sinc = if x.abs() < 1e-12 {
            wn
        } else {
            (wc * x).sin() / (std::f64::consts::PI * x)
        };
        let window = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / order.max(1) as f64).cos();
        h[i] = sinc * window;
    }
    let sum: f64 = h.iter().sum();
    if sum != 0.0 {
        for v in &mut h {
            *v /= sum;
        }
    }
    h
}
