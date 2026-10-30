use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Matrix, Session, Value};
use rustfft::FftPlanner;

use crate::util::{col, matrix_arg, need, positive, real_scalar, row};

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("fft", fft as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("ifft", ifft),
        ("filter", filter),
        ("conv", conv),
        ("freqz", freqz),
        ("butter", butter),
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

fn filter(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let b = matrix_arg(args, 0)?.reals()?;
    let a = matrix_arg(args, 1)?.reals()?;
    let x = matrix_arg(args, 2)?.reals()?;
    if a.is_empty() || a[0] == 0.0 {
        return Err(MatError::new("filter denominator a(1) must be non-zero"));
    }
    let y = direct_form(&b, &a, &x);
    Ok(col(y))
}

pub fn direct_form(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let a0 = a[0];
    let mut y = vec![0.0; x.len()];
    for n in 0..x.len() {
        let mut acc = 0.0;
        for (k, bk) in b.iter().enumerate() {
            if n >= k {
                acc += bk * x[n - k];
            }
        }
        for k in 1..a.len() {
            if n >= k {
                acc -= a[k] * y[n - k];
            }
        }
        y[n] = acc / a0;
    }
    y
}

fn conv(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let a = matrix_arg(args, 0)?.reals()?;
    let b = matrix_arg(args, 1)?.reals()?;
    if a.is_empty() || b.is_empty() {
        return Ok(row(Vec::new()));
    }
    let mut y = vec![0.0; a.len() + b.len() - 1];
    for (i, av) in a.iter().enumerate() {
        for (j, bv) in b.iter().enumerate() {
            y[i + j] += av * bv;
        }
    }
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

fn butter(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    if session.nargout < 2 {
        return Err(MatError::new("butter needs two outputs: [b, a] = butter(n, wn)"));
    }
    let order = positive(real_scalar(args, 0)?)?;
    let wn = real_scalar(args, 1)?;
    if order == 0 || !(0.0..1.0).contains(&wn) {
        return Err(MatError::new("butter order must be >= 1 and wn must be in (0, 1)"));
    }
    let (b, a) = butterworth(order, wn)?;
    Ok(Value::Multi(vec![row(b), row(a)]))
}

pub fn butterworth(order: usize, wn: f64) -> MatResult<(Vec<f64>, Vec<f64>)> {
    let warped = 2.0 * (std::f64::consts::PI * wn / 2.0).tan();
    let mut analog = Vec::with_capacity(order);
    for k in 0..order {
        let m = -((order as i32) - 1) + 2 * k as i32;
        let angle = std::f64::consts::PI * m as f64 / (2.0 * order as f64);
        analog.push(-Complex64::from_polar(1.0, angle) * warped);
    }
    let fs2 = Complex64::from(2.0);
    let mut den = Complex64::from(1.0);
    for pole in &analog {
        den *= fs2 - pole;
    }
    let gain = (Complex64::from(warped.powi(order as i32)) / den).re;
    let zeros = vec![Complex64::from(-1.0); order];
    let poles: Vec<Complex64> = analog
        .into_iter()
        .map(|pole| (fs2 + pole) / (fs2 - pole))
        .collect();
    let b = real_poly_from_roots(&zeros, gain);
    let a = real_poly_from_roots(&poles, 1.0);
    if a.is_empty() || a[0].abs() < 1e-15 {
        return Err(MatError::new("butter design failed"));
    }
    let a0 = a[0];
    Ok((
        b.into_iter().map(|v| v / a0).collect(),
        a.into_iter().map(|v| v / a0).collect(),
    ))
}

fn real_poly_from_roots(roots: &[Complex64], gain: f64) -> Vec<f64> {
    let mut coeff = vec![Complex64::from(gain)];
    for root in roots {
        let mut next = vec![Complex64::new(0.0, 0.0); coeff.len() + 1];
        for (i, c) in coeff.iter().enumerate() {
            next[i] += c;
            next[i + 1] -= c * root;
        }
        coeff = next;
    }
    coeff.into_iter().map(|c| c.re).collect()
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
