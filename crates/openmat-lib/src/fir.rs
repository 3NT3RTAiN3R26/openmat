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

pub(crate) fn fir1(session: &mut Session, args: &[Value]) -> MatResult<Value> {
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

/// Composite Gauss-Legendre quadrature of a smooth `f` over `[x1, x2]`.
fn integrate(f: impl Fn(f64) -> f64, x1: f64, x2: f64) -> f64 {
    const NODES: [f64; 5] = [0.0, 0.5384693101056831, -0.5384693101056831, 0.906179845938664, -0.906179845938664];
    const WEIGHTS: [f64; 5] = [0.5688888888888889, 0.47862867049936647, 0.47862867049936647, 0.23692688505618908, 0.23692688505618908];
    let panels = (((x2 - x1) / 0.01).ceil() as usize).max(1);
    let width = (x2 - x1) / panels as f64;
    let mut total = 0.0;
    for panel in 0..panels {
        let mid = x1 + (panel as f64 + 0.5) * width;
        for (node, weight) in NODES.iter().zip(WEIGHTS) {
            total += weight * f(mid + node * width / 2.0);
        }
    }
    total * width / 2.0
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

/// `integral of sin(c * pi * x)` over `[x1, x2]`.
fn sine_integral(c: f64, x1: f64, x2: f64) -> f64 {
    ((c * PI * x1).cos() - (c * PI * x2).cos()) / (c * PI)
}

/// `integral of x * sin(c * pi * x)` over `[x1, x2]`.
fn x_sine_integral(c: f64, x1: f64, x2: f64) -> f64 {
    let w = c * PI;
    let antiderivative = |x: f64| -x * (w * x).cos() / w + (w * x).sin() / (w * w);
    antiderivative(x2) - antiderivative(x1)
}

#[derive(Clone, Copy, PartialEq)]
enum LsKind {
    Symmetric,
    Hilbert,
    Differentiator,
}

pub(crate) fn firls(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let mut kind = LsKind::Symmetric;
    let mut numeric = Vec::new();
    for arg in args {
        match arg {
            Value::Char(option) => {
                kind = match option.to_ascii_lowercase().as_str() {
                    "hilbert" => LsKind::Hilbert,
                    "differentiator" => LsKind::Differentiator,
                    other => return Err(MatError::new(format!("firls does not know the option '{other}'"))),
                }
            }
            other => numeric.push(other),
        }
    }
    if numeric.len() < 3 || numeric.len() > 4 {
        return Err(MatError::new("firls takes the order, F, A, and optionally W"));
    }
    let mut order = positive(numeric[0].real_scalar()?)?;
    if order == 0 {
        return Err(MatError::new("firls order must be at least 1"));
    }
    let f = numeric[1].clone().into_matrix()?.reals()?;
    let a = numeric[2].clone().into_matrix()?.reals()?;
    let bands = f.len() / 2;
    if f.len() < 2 || f.len() % 2 != 0 || a.len() != f.len() {
        return Err(MatError::new("firls F and A must have the same, even length"));
    }
    if f.windows(2).any(|pair| pair[0] > pair[1]) || f.iter().any(|v| *v < 0.0 || *v > 1.0) {
        return Err(MatError::new("firls F must not decrease and must lie between 0 and 1"));
    }
    let weights = match numeric.get(3) {
        Some(value) => {
            let w = (*value).clone().into_matrix()?.reals()?;
            if w.len() != bands {
                return Err(MatError::new("firls W needs one weight per band"));
            }
            w
        }
        None => vec![1.0; bands],
    };
    // A symmetric filter of odd order has no gain at Nyquist, so MATLAB raises the order.
    if kind == LsKind::Symmetric && order % 2 == 1 && f[f.len() - 1] == 1.0 && a[a.len() - 1] != 0.0 {
        order += 1;
        session.writeln(
            "Warning: Odd order symmetric FIR filters must have a gain of zero at the Nyquist frequency. The order is being increased by one.",
        );
    }
    let odd = order % 2 == 1;
    let half = order / 2;
    let sine = kind != LsKind::Symmetric;
    let basis: Vec<f64> = match (sine, odd) {
        (false, false) => (0..=half).map(|k| k as f64).collect(),
        (false, true) | (true, true) => (1..=half + 1).map(|k| k as f64 - 0.5).collect(),
        (true, false) => (1..=half).map(|k| k as f64).collect(),
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
        let weight = weights[band];
        let desired = |x: f64| slope * x + offset;
        let basis_function = |c: f64, x: f64| if sine { (c * PI * x).sin() } else { (c * PI * x).cos() };
        for (k, ck) in basis.iter().enumerate() {
            for (l, cl) in basis.iter().enumerate() {
                normal[(k, l)] += weight
                    * match (kind, sine) {
                        (LsKind::Differentiator, _) => {
                            integrate(|x| basis_function(*ck, x) * basis_function(*cl, x) / (x * x), x1, x2)
                        }
                        (_, false) => 0.5 * (cosine_integral(ck - cl, x1, x2) + cosine_integral(ck + cl, x1, x2)),
                        (_, true) => 0.5 * (cosine_integral(ck - cl, x1, x2) - cosine_integral(ck + cl, x1, x2)),
                    };
            }
            rhs[k] += weight
                * match (kind, sine) {
                    (LsKind::Differentiator, _) => {
                        integrate(|x| desired(x) * basis_function(*ck, x) / (x * x), x1, x2)
                    }
                    (_, false) => slope * x_cosine_integral(*ck, x1, x2) + offset * cosine_integral(*ck, x1, x2),
                    (_, true) => slope * x_sine_integral(*ck, x1, x2) + offset * sine_integral(*ck, x1, x2),
                };
        }
    }
    let solution = normal
        .lu()
        .solve(&rhs)
        .ok_or_else(|| MatError::new("firls cannot solve for these bands, the specified frequencies leave the response undetermined"))?;
    let mut h = vec![0.0; order + 1];
    match (sine, odd) {
        (false, true) => {
            for k in 1..=half + 1 {
                h[half + k] = solution[k - 1] / 2.0;
                h[half + 1 - k] = solution[k - 1] / 2.0;
            }
        }
        (false, false) => {
            h[half] = solution[0];
            for k in 1..=half {
                h[half - k] = solution[k] / 2.0;
                h[half + k] = solution[k] / 2.0;
            }
        }
        (true, _) => {
            // The Hilbert amplitude is `+sum b sin`, the differentiator amplitude is `-sum b sin`.
            let sign = if kind == LsKind::Differentiator { -1.0 } else { 1.0 };
            if odd {
                for k in 1..=half + 1 {
                    h[half + k] = sign * solution[k - 1] / 2.0;
                    h[half + 1 - k] = -sign * solution[k - 1] / 2.0;
                }
            } else {
                for k in 1..=half {
                    h[half + k] = sign * solution[k - 1] / 2.0;
                    h[half - k] = -sign * solution[k - 1] / 2.0;
                }
            }
        }
    }
    Ok(row(h))
}

/// `(B, G)` of a Savitzky-Golay design. Row `i` of `B` smooths sample `i` of a frame by a weighted
/// polynomial fit, and the columns of `G` give the polynomial coefficients.
fn sgolay_matrices(order: usize, frame: usize, weights: Option<&[f64]>) -> MatResult<(DMatrix<f64>, DMatrix<f64>)> {
    if frame.is_multiple_of(2) {
        return Err(MatError::new("Frame length must be an odd number."));
    }
    if order >= frame {
        return Err(MatError::new("Polynomial order must be less than the frame length."));
    }
    if weights.is_some_and(|w| w.len() != frame) {
        return Err(MatError::new("The weights must have one element per frame sample."));
    }
    let middle = ((frame - 1) / 2) as f64;
    let design = DMatrix::from_fn(frame, order + 1, |i, j| (i as f64 - middle).powi(j as i32));
    let diagonal = DMatrix::from_diagonal(&DVector::from_fn(frame, |i, _| weights.map_or(1.0, |w| w[i])));
    let inverse = (design.transpose() * &diagonal * &design)
        .try_inverse()
        .ok_or_else(|| MatError::new("sgolay design matrix is singular"))?;
    let g = &design * inverse;
    let b = &g * design.transpose() * diagonal;
    Ok((b, g))
}

fn sgolay(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    if args.len() > 3 {
        return Err(MatError::new("sgolay takes the order, the frame length and optionally the weights"));
    }
    let weights = match args.get(2) {
        Some(value) => Some((*value).clone().into_matrix()?.reals()?),
        None => None,
    };
    let (b, g) = sgolay_matrices(positive(real_scalar(args, 0)?)?, positive(real_scalar(args, 1)?)?, weights.as_deref())?;
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![Value::Matrix(from_dmatrix(&b)), Value::Matrix(from_dmatrix(&g))]))
    } else {
        Ok(Value::Matrix(from_dmatrix(&b)))
    }
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
    let weights = match args.get(3) {
        Some(value) => {
            let w = (*value).clone().into_matrix()?.reals()?;
            (!w.is_empty()).then_some(w)
        }
        None => None,
    };
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
        return Err(MatError::new("The length of the input must be >= frame length."));
    }
    let (b, _) = sgolay_matrices(order, frame, weights.as_deref())?;
    let result = along(&x, dim, length, |seq| {
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
        // MATLAB raises an odd order that would need a gain at Nyquist, so this is the order 6 delta.
        assert_close(&reals("h = firls(5, [0 1], [1 1]);", "h"), &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 1e-12);
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
            "h = firls(10, [0 1], [1 1], 'nope');",
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

    /// Symmetric filter of length `n` from its first `ceil(n / 2)` taps.
    fn sym(first: &[f64], n: usize) -> Vec<f64> {
        (0..n).map(|i| first[i.min(n - 1 - i)]).collect()
    }

    /// Antisymmetric filter of length `n` from its first `floor(n / 2)` taps.
    fn anti(first: &[f64], n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| match i.cmp(&(n - 1 - i)) {
                std::cmp::Ordering::Less => first[i],
                std::cmp::Ordering::Equal => 0.0,
                std::cmp::Ordering::Greater => -first[n - 1 - i],
            })
            .collect()
    }

    #[test]
    fn fir1_matches_matlab_reference() {
        let tol = 1e-12;
        let check = |src: &str, want: Vec<f64>| assert_close(&reals(src, "h"), &want, tol);
        check("h = fir1(20, 0.4);", sym(&[-1.24995293746598e-18, -0.0034552528812046, -0.0039335848491553, 0.00722110437359731, 0.0201145189915256, -8.43718232789538e-18, -0.0517318083030213, -0.0506430237108573, 0.0855040614772554, 0.296517071230397, 0.400813827342926], 21));
        check("h = fir1(20, 0.4, 'high');", sym(&[-1.87381605139963e-18, 0.00345320111202038, 0.00393124904090782, -0.00721681639817576, -0.0201025747571551, 2.95126028095442e-17, 0.0517010893560239, 0.0506129512967625, -0.0854532881751277, -0.296340995958843, 0.600863729174231], 21));
        check("h = fir1(30, [0.3 0.5]);", sym(&[-0.0034712603232985, -0.0012303905105492, 0.00392468803392467, 0.00432922874096561, -0.0013119659603504, 2.47093318428705e-18, 0.00274746659032512, -0.0190312164430249, -0.0357885744370809, 0.0217472521292618, 0.100232641835244, 0.0405425076522155, -0.129523244309934, -0.148597147502686, 0.0615269690942701, 0.20447411119448], 31));
        check("h = fir1(30, [0.3 0.5], 'stop');", sym(&[0.00338427585271603, 0.0011995588075935, -0.00382634135893145, -0.00422074489504493, 0.0012790901014823, -1.44540922842305e-17, -0.00267861928285052, 0.0185543232881883, 0.0348917675397656, -0.0212022992772989, -0.0977209652471751, -0.0395265744649106, 0.126277590056116, 0.14487353042942, -0.0599851974220942, 0.797401211746049], 31));
        check("h = fir1(40, [0.2 0.4 0.6 0.8]);", sym(&[-1.241342423339e-18, 3.1054555171778e-18, 0.00131069756802382, 8.98610884564657e-18, 0.0102273175192479, -1.66596594848117e-17, -0.018774865701516, 9.14691164376e-18, -0.00763013649958101, -3.07027728207262e-17, -8.37906135753825e-18, 1.05641322300459e-17, 0.0196236737696043, 1.03415916086254e-17, 0.131671821884549, -6.71309540905707e-17, -0.222310492705138, 1.04942605291009e-16, -0.112479235963689, 8.2392100543769e-17, 0.398052752890296], 41));
        check("h = fir1(40, [0.2 0.4 0.6 0.8], 'DC-1');", sym(&[-1.87362718271024e-18, 5.35684204204959e-18, -0.00131887250747787, -1.26590184286633e-17, -0.0102911062250003, 4.13557812562204e-17, 0.0188919662395136, -2.41603996205367e-17, 0.00767772635206253, 9.75608488289989e-17, -1.26469834832941e-17, 1.27560260155898e-17, -0.0197460684004072, 1.04060930522242e-17, -0.132493070964405, 1.01324483897879e-16, 0.223697063384904, -6.59982136411584e-17, 0.113180779146703, -5.5270658358595e-17, 0.600803165948215], 41));
        check("h = fir1(10, 0.3, hann(11));", sym(&[0.0, -0.00456836031765592, 0.011586050263709, 0.101327784574171, 0.238235909680666, 0.30683723159822], 11));
        check("h = fir1(10, 0.3, 'noscale');", sym(&[-0.00509295817894065, -0.00785119590355814, 0.0130446665354303, 0.10325354022033, 0.234894579609322, 0.3], 11));
        check("h = fir1(10, [0.2 0.5], 'noscale');", sym(&[0.00509295817894065, -0.00785119590355815, -0.082360783096504, -0.10325354022033, 0.119684766259245, 0.3], 11));
        check("h = fir1(4, 0.5, 'noscale', ones(5, 1));", sym(&[1.94908591625969e-17, std::f64::consts::FRAC_1_PI, 0.5], 5));
        check("h = fir1(21, 0.4, 'high');", sym(&[-0.00219614298361115, -2.30113634437792e-18, 0.00513426850306929, 0.00556999291260534, -0.00930229095578301, -0.0238825375644789, 3.29598168783754e-17, 0.0551918181220935, 0.0523325904636877, -0.0864996713557611, -0.29634298946873, 0.598491666888348], 23));
        check("h = fir1(11, [0.2 0.4], 'stop');", sym(&[-0.00716733669579257, 9.69406716704764e-18, 0.0416601445442943, 0.0967590453931997, 0.048855884322508, -0.119077866311874, 0.877940257495329], 13));
        check("h = fir1(21, 0.4);", sym(&[0.00142588943608162, -0.00197857796770562, -0.00569749903363954, 3.94904349130186e-18, 0.0173275031798074, 0.0172048261641859, -0.0267147361261761, -0.0666185973851324, 1.36816304572834e-17, 0.192678280093767, 0.372372911638811], 22));
    }

    #[test]
    fn firls_matches_matlab_reference() {
        let tol = 1e-9;
        let check = |src: &str, want: Vec<f64>, tolerance: f64| assert_close(&reals(src, "h"), &want, tolerance);
        check("h = firls(20, [0 0.4 0.5 1], [1 1 0 0]);", sym(&[0.0165089897941586, 0.00359581318499373, -0.0253805457464427, -0.0156447205828506, 0.034460958749132, 0.0391469193006457, -0.0423447853611655, -0.0899604804687433, 0.047708412170353, 0.312673797755867, 0.450390116429352], 21), tol);
        check("h = firls(21, [0 0.4 0.5 1], [1 1 0 0]);", sym(&[0.01203197679352, 0.0134054339683914, -0.0138544456250502, -0.0273025720802726, 0.0105923469382263, 0.0482481408362641, 0.00307679604504862, -0.0829348514011069, -0.0452264372225747, 0.17970256245735, 0.411352946156759], 22), tol);
        check("h = firls(20, [0 0.4 0.5 1], [1 1 0 0], [1 100]);", sym(&[-0.00734926399308325, -0.0244956785292908, -0.0253089876957879, 0.00968252879961858, 0.0454238059068928, 0.0183251237683959, -0.0615889814561672, -0.0761402818865987, 0.0724122370058901, 0.30782193141912, 0.423762252114697], 21), tol);
        check("h = firls(20, [0 0.3 0.4 0.6 0.7 1], [0 0 1 1 0 0]);", sym(&[0.0349249458233189, 0.0, -0.0370641891927234, 0.0, -0.0209236117413747, 0.0, 0.136805534741694, 0.0, -0.256227903132472, 0.0, 0.307102999793201], 21), tol);
        check("h = firls(21, [0 1], [0 1], 'hilbert');", anti(&[-0.000919013003558713, 0.00112267239492889, -0.00140236932376945, 0.00180126548697482, -0.00239813452407909, 0.00334946061627552, -0.00500351524159692, 0.00827111703202757, -0.016211389382774, 0.0450316371743723, -0.405284734569351], 22), tol);
        check("h = firls(21, [0.1 0.9], [1 1], 'hilbert');", anti(&[-0.00596738583776827, -0.00642280381985289, -0.0147899627262529, -0.0164615240006706, -0.0301028214714402, -0.0351354470745287, -0.0576531278171409, -0.0735413995376529, -0.121565538236354, -0.201830105256576, -0.638980451297301], 22), tol);
        // The 1/w^2 weighting of the differentiator agrees with MATLAB to about 1e-4.
        check("h = firls(20, [0 0.9], [0 0.9 * pi], 'differentiator');", anti(&[-0.00873931694227708, 0.024709126785899, -0.0419434888437881, 0.0646326289229113, -0.0951291257663463, 0.136989267168566, -0.197258610429048, 0.292342708226955, -0.471988128269258, 0.985756388380762], 21), 2e-4);
    }

    #[test]
    fn sgolay_matches_matlab_reference() {
        let tol = 1e-12;
        let rows = |name: &str, src: &str| -> Vec<Vec<f64>> {
            let session = exec(src).unwrap();
            let m = session.get(name).unwrap().clone().into_matrix().unwrap();
            (0..m.rows).map(|r| (0..m.cols).map(|c| m.get(r, c).re).collect()).collect()
        };
        let b = rows("B", "B = sgolay(3, 7);");
        assert_close(&b[0], &[0.928571428571429, 0.190476190476191, -0.0952380952380951, -0.0952380952380952, 0.0238095238095238, 0.0952380952380952, -0.0476190476190476], tol);
        assert_close(&b[3], &[-0.0952380952380952, 0.142857142857143, 0.285714285714286, 0.333333333333333, 0.285714285714286, 0.142857142857143, -0.0952380952380952], tol);
        let g = rows("G", "[B, G] = sgolay(2, 5);");
        assert_close(&g[0], &[-0.0857142857142857, -0.2, 0.142857142857143], tol);
        assert_close(&g[2], &[0.485714285714286, 0.0, -0.142857142857143], tol);
        let b = rows("B", "B = sgolay(2, 5, [1 2 3 2 1]);");
        assert_close(&b[0], &[0.8, 0.466666666666667, -0.2, -0.2, 0.133333333333333], 1e-9);
        assert_close(&b[2], &[-0.0666666666666667, 0.266666666666667, 0.6, 0.266666666666667, -0.0666666666666667], 1e-9);
        let x = "x = [1 5 2 8 3 9 4 7 2 6];";
        assert_close(
            &reals(&format!("{x} y = sgolayfilt(x, 2, 5);"), "y"),
            &[1.11428571428571, 3.74285714285714, 5.08571428571429, 4.4, 6.77142857142857, 5.48571428571428, 7.0, 4.17142857142857, 4.08571428571428, 5.42857142857142],
            1e-9,
        );
        assert_close(
            &reals(&format!("{x} y = sgolayfilt(x, 1, 3, [1 2 1]);"), "y"),
            &[2.75, 3.25, 4.25, 5.25, 5.75, 6.25, 6.0, 5.0, 4.25, 3.75],
            1e-9,
        );
        let err = exec("b = sgolay(2, 4);").err().unwrap().to_string();
        assert!(err.contains("Frame length must be an odd number."), "{err}");
        let err = exec("y = sgolayfilt([1 2 3], 1, 5);").err().unwrap().to_string();
        assert!(err.contains("The length of the input must be >= frame length."), "{err}");
    }

    #[test]
    fn pulse_shapes_match_matlab_reference() {
        let tol = 1e-12;
        assert_close(&reals("h = gaussdesign(0.5, 3, 2);", "h"), &sym(&[8.24920015562788e-08, 0.000604386797959435, 0.125967017250466, 0.746857026919146], 7), 1e-9);
        assert_close(
            &reals("h = gaussdesign(0.3, 4, 3);", "h"),
            &sym(&[1.06245129748805e-05, 0.000243643557567244, 0.00316116404081411, 0.0232052009543039, 0.0963762078974151, 0.226464709448811, 0.301076899176227], 13),
            1e-9,
        );
        let root = reals("h = rcosdesign(0.25, 6, 4);", "h");
        assert_close(&root[..13], &[-0.0187733440045221, 0.003013558666422, 0.0326772345462546, 0.0470935833402524, 0.0265495177022908, -0.0275222240300177, -0.0852248750170832, -0.0994474359919262, -0.0321472673983147, 0.119037148179254, 0.311176411555479, 0.472003895565433, 0.534632070178156], tol);
        let normal = reals("h = rcosdesign(0.25, 6, 4, 'normal');", "h");
        assert_close(&normal[..13], &[0.0, 0.0263937423242705, 0.0447797538925778, 0.0379816510540092, 0.0, -0.055344691535842, -0.0959566154840953, -0.084866033011885, 0.0, 0.150103902071082, 0.324323667501572, 0.463727589062769, 0.516957035194441], tol);
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
