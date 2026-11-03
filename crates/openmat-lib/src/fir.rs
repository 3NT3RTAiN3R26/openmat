use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Session, Value};

use crate::numeric::from_dmatrix;
use crate::sigproc::{along, default_dim};
use crate::util::{matrix_arg, need, positive, real_scalar, row};

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("fir1", fir1 as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("firls", firls),
        ("sgolay", sgolay),
        ("sgolayfilt", sgolayfilt),
        ("gaussdesign", gaussdesign),
        ("rcosdesign", rcosdesign),
    ] {
        session.register_builtin(name, func);
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

fn hamming(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            if len > 1 {
                0.54 - 0.46 * (2.0 * PI * i as f64 / (len - 1) as f64).cos()
            } else {
                1.0
            }
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Low,
    High,
    Pass,
    Stop,
    DcZero,
    DcOne,
}

fn fir1(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let mut order = positive(real_scalar(args, 0)?)?;
    if order == 0 {
        return Err(MatError::new("fir1 order must be at least 1"));
    }
    let wn = matrix_arg(args, 1)?.reals()?;
    let mut kind = None;
    let mut scale = true;
    let mut window = None;
    for arg in &args[2..] {
        match arg {
            Value::Char(option) => match option.to_ascii_lowercase().as_str() {
                "low" | "lowpass" => kind = Some(Kind::Low),
                "high" | "highpass" => kind = Some(Kind::High),
                "bandpass" => kind = Some(Kind::Pass),
                "stop" | "bandstop" => kind = Some(Kind::Stop),
                "dc-0" => kind = Some(Kind::DcZero),
                "dc-1" => kind = Some(Kind::DcOne),
                "scale" => scale = true,
                "noscale" => scale = false,
                other => return Err(MatError::new(format!("fir1 does not know the option '{other}'"))),
            },
            other => window = Some(other.clone().into_matrix()?.reals()?),
        }
    }
    let increasing = wn.windows(2).all(|pair| pair[0] < pair[1]);
    if wn.is_empty() || !increasing || wn.iter().any(|w| *w <= 0.0 || *w >= 1.0) {
        return Err(MatError::new("fir1 Wn must increase strictly and lie between 0 and 1"));
    }
    let kind = kind.unwrap_or(match wn.len() {
        1 => Kind::Low,
        2 => Kind::Pass,
        _ => Kind::DcZero,
    });
    let count_ok = match kind {
        Kind::Low | Kind::High => wn.len() == 1,
        Kind::Pass | Kind::Stop => wn.len() == 2,
        Kind::DcZero | Kind::DcOne => true,
    };
    if !count_ok {
        return Err(MatError::new("fir1 Wn has the wrong number of elements for this filter type"));
    }
    let first_is_pass = matches!(kind, Kind::Low | Kind::Stop | Kind::DcOne);
    let segments = wn.len() + 1;
    let gains: Vec<f64> = (0..segments)
        .map(|i| f64::from((i % 2 == 0) == first_is_pass))
        .collect();
    if gains[segments - 1] == 1.0 && order % 2 == 1 {
        order += 1;
        session.writeln(
            "Warning: Odd order symmetric FIR filters must have a gain of zero at the Nyquist frequency. The order is being increased by one.",
        );
    }
    let window = match window {
        Some(w) if w.len() == order + 1 => w,
        Some(_) => return Err(MatError::new("fir1 window must have order + 1 elements")),
        None => hamming(order + 1),
    };
    let mut edges = vec![0.0];
    edges.extend_from_slice(&wn);
    edges.push(1.0);
    let middle = order as f64 / 2.0;
    let mut h: Vec<f64> = (0..=order)
        .map(|i| {
            let k = i as f64 - middle;
            let ideal: f64 = (0..segments)
                .filter(|s| gains[*s] == 1.0)
                .map(|s| edges[s + 1] * sinc(edges[s + 1] * k) - edges[s] * sinc(edges[s] * k))
                .sum();
            ideal * window[i]
        })
        .collect();
    if scale {
        let band = gains.iter().position(|g| *g == 1.0).unwrap_or(0);
        let (low, high) = (edges[band], edges[band + 1]);
        let f0 = if low == 0.0 {
            0.0
        } else if high == 1.0 {
            1.0
        } else {
            (low + high) / 2.0
        };
        let response: Complex64 = h
            .iter()
            .enumerate()
            .map(|(i, v)| Complex64::from_polar(*v, -PI * f0 * i as f64))
            .sum();
        if response.norm() > 0.0 {
            for v in &mut h {
                *v /= response.norm();
            }
        }
    }
    Ok(row(h))
}

/// `integral of cos(c * pi * x)` over `[x1, x2]`.
fn cosine_integral(c: f64, x1: f64, x2: f64) -> f64 {
    if c == 0.0 {
        x2 - x1
    } else {
        ((c * PI * x2).sin() - (c * PI * x1).sin()) / (c * PI)
    }
}

/// `integral of x * cos(c * pi * x)` over `[x1, x2]`.
fn x_cosine_integral(c: f64, x1: f64, x2: f64) -> f64 {
    if c == 0.0 {
        return (x2 * x2 - x1 * x1) / 2.0;
    }
    let w = c * PI;
    let antiderivative = |x: f64| x * (w * x).sin() / w + (w * x).cos() / (w * w);
    antiderivative(x2) - antiderivative(x1)
}

fn firls(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    if args.iter().skip(3).any(|arg| matches!(arg, Value::Char(_))) {
        return Err(MatError::new("firls types 'hilbert' and 'differentiator' are not supported"));
    }
    let order = positive(real_scalar(args, 0)?)?;
    if order == 0 {
        return Err(MatError::new("firls order must be at least 1"));
    }
    let f = matrix_arg(args, 1)?.reals()?;
    let a = matrix_arg(args, 2)?.reals()?;
    let bands = f.len() / 2;
    if f.len() < 2 || f.len() % 2 != 0 || a.len() != f.len() {
        return Err(MatError::new("firls F and A must have the same, even length"));
    }
    if f.windows(2).any(|pair| pair[0] > pair[1]) || f.iter().any(|v| *v < 0.0 || *v > 1.0) {
        return Err(MatError::new("firls F must not decrease and must lie between 0 and 1"));
    }
    let weights = if args.len() >= 4 {
        let w = matrix_arg(args, 3)?.reals()?;
        if w.len() != bands {
            return Err(MatError::new("firls W needs one weight per band"));
        }
        w
    } else {
        vec![1.0; bands]
    };
    let odd = order % 2 == 1;
    let half = order / 2;
    let basis: Vec<f64> = if odd {
        (1..=half + 1).map(|k| k as f64 - 0.5).collect()
    } else {
        (0..=half).map(|k| k as f64).collect()
    };
    let size = basis.len();
    let mut normal = DMatrix::<f64>::zeros(size, size);
    let mut rhs = DVector::<f64>::zeros(size);
    for band in 0..bands {
        let (x1, x2) = (f[2 * band], f[2 * band + 1]);
        if x2 <= x1 {
            continue;
        }
        let slope = (a[2 * band + 1] - a[2 * band]) / (x2 - x1);
        let offset = a[2 * band] - slope * x1;
        for (k, ck) in basis.iter().enumerate() {
            for (l, cl) in basis.iter().enumerate() {
                normal[(k, l)] += weights[band]
                    * 0.5
                    * (cosine_integral(ck - cl, x1, x2) + cosine_integral(ck + cl, x1, x2));
            }
            rhs[k] += weights[band] * (slope * x_cosine_integral(*ck, x1, x2) + offset * cosine_integral(*ck, x1, x2));
        }
    }
    let solution = normal
        .lu()
        .solve(&rhs)
        .ok_or_else(|| MatError::new("firls cannot solve for these bands, the specified frequencies leave the response undetermined"))?;
    let mut h = vec![0.0; order + 1];
    if odd {
        for k in 1..=half + 1 {
            h[half + k] = solution[k - 1] / 2.0;
            h[half + 1 - k] = solution[k - 1] / 2.0;
        }
    } else {
        h[half] = solution[0];
        for k in 1..=half {
            h[half - k] = solution[k] / 2.0;
            h[half + k] = solution[k] / 2.0;
        }
    }
    Ok(row(h))
}

/// The projection matrix whose row `i` smooths sample `i` of a frame by a polynomial fit.
fn sgolay_matrix(order: usize, frame: usize) -> MatResult<DMatrix<f64>> {
    if frame.is_multiple_of(2) {
        return Err(MatError::new("sgolay frame length must be odd"));
    }
    if order >= frame {
        return Err(MatError::new("sgolay order must be less than the frame length"));
    }
    let middle = ((frame - 1) / 2) as f64;
    let design = DMatrix::from_fn(frame, order + 1, |i, j| (i as f64 - middle).powi(j as i32));
    let inverse = (design.transpose() * &design)
        .try_inverse()
        .ok_or_else(|| MatError::new("sgolay design matrix is singular"))?;
    Ok(&design * inverse * design.transpose())
}

fn sgolay(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    if args.len() > 2 {
        return Err(MatError::new("sgolay weights are not supported"));
    }
    let b = sgolay_matrix(positive(real_scalar(args, 0)?)?, positive(real_scalar(args, 1)?)?)?;
    Ok(Value::Matrix(from_dmatrix(&b)))
}

fn smooth(x: &[f64], b: &DMatrix<f64>, frame: usize) -> Vec<f64> {
    let n = x.len();
    let m = (frame - 1) / 2;
    let apply = |row: usize, start: usize| -> f64 { (0..frame).map(|j| b[(row, j)] * x[start + j]).sum() };
    (0..n)
        .map(|i| {
            if i < m {
                apply(i, 0)
            } else if i >= n - m {
                apply(i - (n - frame), n - frame)
            } else {
                apply(m, i - m)
            }
        })
        .collect()
}

fn sgolayfilt(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let x = matrix_arg(args, 0)?;
    let order = positive(real_scalar(args, 1)?)?;
    let frame = positive(real_scalar(args, 2)?)?;
    if args.len() >= 4 && matrix_arg(args, 3)?.numel() > 0 {
        return Err(MatError::new("sgolayfilt weights are not supported"));
    }
    let dim = if args.len() >= 5 {
        positive(real_scalar(args, 4)?)?
    } else {
        default_dim(&x)
    };
    if dim != 1 && dim != 2 {
        return Err(MatError::new("dimension must be 1 or 2"));
    }
    let length = if dim == 1 { x.rows } else { x.cols };
    if frame > length {
        return Err(MatError::new("sgolayfilt frame length must not exceed the signal length"));
    }
    let b = sgolay_matrix(order, frame)?;
    let result = along(&x, dim, |seq| {
        let re: Vec<f64> = seq.iter().map(|z| z.re).collect();
        let im: Vec<f64> = seq.iter().map(|z| z.im).collect();
        smooth(&re, &b, frame)
            .into_iter()
            .zip(smooth(&im, &b, frame))
            .map(|(r, i)| Complex64::new(r, i))
            .collect()
    })?;
    Ok(Value::Matrix(result))
}

/// `span * sps`, which must be a positive even integer.
fn pulse_length(span: f64, sps: f64, name: &str) -> MatResult<usize> {
    let n = span * sps;
    if n < 2.0 || n.fract() != 0.0 || n % 2.0 != 0.0 {
        return Err(MatError::new(format!("{name}: span * sps must be a positive even integer")));
    }
    Ok(n as usize)
}

fn gaussdesign(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let bt = real_scalar(args, 0)?;
    let span = real_scalar(args, 1)?;
    let sps = real_scalar(args, 2)?;
    if bt <= 0.0 || sps < 1.0 {
        return Err(MatError::new("gaussdesign needs bt > 0 and sps >= 1"));
    }
    let n = pulse_length(span, sps, "gaussdesign")?;
    let alpha = (std::f64::consts::LN_2 / 2.0).sqrt() / bt;
    let mut h: Vec<f64> = (0..=n)
        .map(|i| {
            let t = (i as f64 - n as f64 / 2.0) / sps;
            PI.sqrt() / alpha * (-(PI * t / alpha).powi(2)).exp()
        })
        .collect();
    let total: f64 = h.iter().sum();
    for v in &mut h {
        *v /= total;
    }
    Ok(row(h))
}

fn raised_cosine(t: f64, beta: f64) -> f64 {
    if beta > 0.0 && ((2.0 * beta * t).abs() - 1.0).abs() < 1e-12 {
        return PI / 4.0 * sinc(1.0 / (2.0 * beta));
    }
    sinc(t) * (PI * beta * t).cos() / (1.0 - (2.0 * beta * t).powi(2))
}

fn root_raised_cosine(t: f64, beta: f64) -> f64 {
    if t == 0.0 {
        return 1.0 - beta + 4.0 * beta / PI;
    }
    if beta > 0.0 && ((4.0 * beta * t).abs() - 1.0).abs() < 1e-12 {
        let angle = PI / (4.0 * beta);
        return beta / std::f64::consts::SQRT_2
            * ((1.0 + 2.0 / PI) * angle.sin() + (1.0 - 2.0 / PI) * angle.cos());
    }
    let numerator = (PI * t * (1.0 - beta)).sin() + 4.0 * beta * t * (PI * t * (1.0 + beta)).cos();
    numerator / (PI * t * (1.0 - (4.0 * beta * t).powi(2)))
}

fn rcosdesign(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let beta = real_scalar(args, 0)?;
    let span = real_scalar(args, 1)?;
    let sps = real_scalar(args, 2)?;
    if !(0.0..=1.0).contains(&beta) || sps < 1.0 {
        return Err(MatError::new("rcosdesign needs 0 <= beta <= 1 and sps >= 1"));
    }
    let root = match args.get(3) {
        None => true,
        Some(Value::Char(shape)) if shape == "sqrt" => true,
        Some(Value::Char(shape)) if shape == "normal" => false,
        Some(_) => return Err(MatError::new("rcosdesign shape must be 'sqrt' or 'normal'")),
    };
    let n = pulse_length(span, sps, "rcosdesign")?;
    let mut h: Vec<f64> = (0..=n)
        .map(|i| {
            let t = (i as f64 - n as f64 / 2.0) / sps;
            if root {
                root_raised_cosine(t, beta)
            } else {
                raised_cosine(t, beta)
            }
        })
        .collect();
    let energy = h.iter().map(|v| v * v).sum::<f64>().sqrt();
    for v in &mut h {
        *v /= energy;
    }
    Ok(row(h))
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

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

    fn assert_close(got: &[f64], want: &[f64], tolerance: f64) {
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < tolerance, "{got:?} vs {want:?}");
        }
    }

    /// Magnitude response of the FIR filter `h` at the normalized frequency `w` (1 is Nyquist).
    fn gain(h: &[f64], w: f64) -> f64 {
        let (re, im) = h.iter().enumerate().fold((0.0, 0.0), |(re, im), (i, v)| {
            (re + v * (PI * w * i as f64).cos(), im - v * (PI * w * i as f64).sin())
        });
        re.hypot(im)
    }

    fn is_symmetric(h: &[f64]) -> bool {
        h.iter().zip(h.iter().rev()).all(|(a, b)| (a - b).abs() < 1e-12)
    }

    #[test]
    fn fir1_lowpass_and_highpass() {
        let h = reals("h = fir1(20, 0.4);", "h");
        assert_eq!(h.len(), 21);
        assert!(is_symmetric(&h));
        assert!((gain(&h, 0.0) - 1.0).abs() < 1e-12);
        assert!(gain(&h, 0.8) < 0.01);
        let h = reals("h = fir1(20, 0.4, 'high');", "h");
        assert!(is_symmetric(&h));
        assert!((gain(&h, 1.0) - 1.0).abs() < 1e-12);
        assert!(gain(&h, 0.0) < 0.01);
    }

    #[test]
    fn fir1_matches_the_windowed_ideal_response() {
        let h = reals("h = fir1(4, 0.5, 'noscale', ones(5, 1));", "h");
        let sinc = |k: f64| if k == 0.0 { 1.0 } else { (PI * 0.5 * k).sin() / (PI * 0.5 * k) };
        let want: Vec<f64> = (0..5).map(|i| 0.5 * sinc(i as f64 - 2.0)).collect();
        assert_close(&h, &want, 1e-12);
        let hamming = reals("h = fir1(10, 0.3, hamming(11));", "h");
        assert_close(&hamming, &reals("h = fir1(10, 0.3);", "h"), 1e-12);
    }

    #[test]
    fn fir1_band_types() {
        let h = reals("h = fir1(30, [0.3 0.5]);", "h");
        assert!((gain(&h, 0.4) - 1.0).abs() < 1e-12);
        assert!(gain(&h, 0.0) < 0.05 && gain(&h, 1.0) < 0.05);
        let h = reals("h = fir1(30, [0.3 0.5], 'stop');", "h");
        assert!((gain(&h, 0.0) - 1.0).abs() < 1e-12);
        assert!(gain(&h, 0.4) < 0.1);
        let h = reals("h = fir1(40, [0.2 0.4 0.6 0.8]);", "h");
        assert!((gain(&h, 0.3) - 1.0).abs() < 1e-12);
        assert!((gain(&h, 0.7) - 1.0).abs() < 0.05);
        assert!(gain(&h, 0.5) < 0.1);
        let h = reals("h = fir1(40, [0.2 0.4 0.6 0.8], 'DC-1');", "h");
        assert!((gain(&h, 0.0) - 1.0).abs() < 1e-12);
        assert!((gain(&h, 0.5) - 1.0).abs() < 0.05);
    }

    #[test]
    fn fir1_raises_odd_orders_that_pass_nyquist() {
        let session = exec("h = fir1(21, 0.4, 'high');").unwrap();
        let h = session.get("h").unwrap().clone().into_matrix().unwrap().reals().unwrap();
        assert_eq!(h.len(), 23);
        assert!(session.output.contains("order is being increased"), "{}", session.output);
        assert_eq!(reals("h = fir1(21, 0.4);", "h").len(), 22);
    }

    #[test]
    fn fir1_rejects_bad_arguments() {
        for src in [
            "h = fir1(0, 0.4);",
            "h = fir1(10, 1.2);",
            "h = fir1(10, [0.5 0.3]);",
            "h = fir1(10, [0.2 0.4], 'high');",
            "h = fir1(10, 0.4, 'nope');",
            "h = fir1(10, 0.3, ones(5, 1));",
        ] {
            assert!(crate::exec(src).is_err(), "{src}");
        }
    }

    #[test]
    fn firls_designs_least_squares_filters() {
        assert_close(&reals("h = firls(6, [0 1], [1 1]);", "h"), &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 1e-12);
        // Orthogonal cosine basis: h = (1/pi) * [2/5, -2/3, 2] mirrored.
        let (a, b, c) = (0.4 / PI, -2.0 / (3.0 * PI), 2.0 / PI);
        assert_close(&reals("h = firls(5, [0 1], [1 1]);", "h"), &[a, b, c, c, b, a], 1e-12);
        for order in [20, 21] {
            let h = reals(&format!("h = firls({order}, [0 0.4 0.5 1], [1 1 0 0]);"), "h");
            assert_eq!(h.len(), order + 1);
            assert!(is_symmetric(&h));
            assert!((gain(&h, 0.2) - 1.0).abs() < 0.05, "{}", gain(&h, 0.2));
            assert!(gain(&h, 0.8) < 0.05, "{}", gain(&h, 0.8));
        }
        let even = reals("h = firls(20, [0 0.4 0.5 1], [1 1 0 0], [1 1]);", "h");
        let heavy = reals("h = firls(20, [0 0.4 0.5 1], [1 1 0 0], [1 100]);", "h");
        let stop_error = |h: &[f64]| (0..20).map(|i| gain(h, 0.5 + 0.025 * i as f64).powi(2)).sum::<f64>();
        assert!(stop_error(&heavy) < stop_error(&even));
        let ramp = reals("h = firls(30, [0 0.5], [0 1]);", "h");
        assert!((gain(&ramp, 0.25) - 0.5).abs() < 0.05);
    }

    #[test]
    fn firls_rejects_bad_arguments() {
        for src in [
            "h = firls(10, [0 0.5 1], [1 1 0]);",
            "h = firls(10, [0 0.5], [1 1 0]);",
            "h = firls(10, [0 1], [1 1], 'hilbert');",
            "h = firls(10, [0.5 0.2], [1 1]);",
            "h = firls(10, [0 0.5 0.6 1], [1 1 0 0], [1]);",
        ] {
            assert!(crate::exec(src).is_err(), "{src}");
        }
    }

    #[test]
    fn savitzky_golay_matrix_matches_known_filters() {
        let b = reals("b = sgolay(2, 5);", "b");
        let center: Vec<f64> = (0..5).map(|j| b[j * 5 + 2]).collect();
        assert_close(&center, &[-3.0 / 35.0, 12.0 / 35.0, 17.0 / 35.0, 12.0 / 35.0, -3.0 / 35.0], 1e-12);
        let b = reals("b = sgolay(1, 3);", "b");
        let first: Vec<f64> = (0..3).map(|j| b[j * 3]).collect();
        assert_close(&first, &[5.0 / 6.0, 1.0 / 3.0, -1.0 / 6.0], 1e-12);
        assert_close(&reals("b = sgolay(0, 3);", "b"), &[1.0 / 3.0; 9], 1e-12);
        let b = reals("b = sgolay(2, 3);", "b");
        assert_close(&b, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 1e-12);
        assert!(crate::exec("b = sgolay(2, 4);").is_err());
        assert!(crate::exec("b = sgolay(5, 5);").is_err());
    }

    #[test]
    fn savitzky_golay_filter_reproduces_polynomials() {
        let src = "k = 1:20; x = k .* k - 3 * k + 2; y = sgolayfilt(x, 2, 5);";
        assert_close(&reals(src, "y"), &reals(src, "x"), 1e-9);
        let noisy = reals("x = [1 5 1 5 1 5 1 5 1 5 1]; y = sgolayfilt(x, 1, 5);", "y");
        assert!((noisy[5] - 3.4).abs() < 1e-9 || (noisy[5] - 2.6).abs() < 1e-9, "{noisy:?}");
        assert_eq!(reals("x = ones(7, 1); y = sgolayfilt(x, 2, 3);", "y").len(), 7);
        assert_close(&reals("x = [1 1 1; 2 2 2; 3 3 3; 4 4 4]; y = sgolayfilt(x, 1, 3);", "y"), &reals("x = [1 1 1; 2 2 2; 3 3 3; 4 4 4];", "x"), 1e-9);
        assert!(crate::exec("y = sgolayfilt([1 2 3], 1, 5);").is_err());
    }

    #[test]
    fn gaussian_pulse() {
        let h = reals("h = gaussdesign(0.3, 4, 8);", "h");
        assert_eq!(h.len(), 33);
        assert!(is_symmetric(&h));
        assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let peak = h.iter().cloned().fold(f64::MIN, f64::max);
        assert_eq!(peak, h[16]);
        let narrow = reals("h = gaussdesign(0.5, 4, 8);", "h");
        assert!(narrow[16] > h[16]);
        assert!(crate::exec("h = gaussdesign(0.3, 3, 3);").is_err());
        assert!(crate::exec("h = gaussdesign(0, 4, 8);").is_err());
    }

    #[test]
    fn raised_cosine_pulses_have_no_intersymbol_interference() {
        let normal = reals("h = rcosdesign(0.25, 12, 4, 'normal');", "h");
        assert_eq!(normal.len(), 49);
        assert!(is_symmetric(&normal));
        let center = 24;
        for m in 1..=5 {
            assert!(normal[center + 4 * m].abs() < 1e-12 * normal[center], "{m}");
        }
        let root = reals("h = rcosdesign(0.25, 20, 4);", "h");
        assert!(is_symmetric(&root));
        assert!((root.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
        let c = root.len() - 1;
        let autocorrelation = |lag: usize| -> f64 { (0..root.len() - lag).map(|i| root[i] * root[i + lag]).sum() };
        assert!(c == 80 && autocorrelation(0) > 0.99);
        for m in 1..=4 {
            assert!(autocorrelation(4 * m).abs() < 5e-3, "{m}: {}", autocorrelation(4 * m));
        }
        let singular = reals("h = rcosdesign(0.5, 8, 2, 'normal');", "h");
        assert!(singular.iter().all(|v| v.is_finite()));
        let singular = reals("h = rcosdesign(0.5, 8, 2);", "h");
        assert!(singular.iter().all(|v| v.is_finite()));
        assert!(crate::exec("h = rcosdesign(1.5, 8, 2);").is_err());
        assert!(crate::exec("h = rcosdesign(0.5, 3, 3);").is_err());
        assert!(crate::exec("h = rcosdesign(0.5, 8, 2, 'other');").is_err());
    }
}
