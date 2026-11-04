//! State-space frequency transforms, minimum-order IIR estimates, elliptic and Bessel
//! prototypes, and the comb filter. The state-space realization follows MATLAB: a
//! controllable-canonical cascade of the analog prototype, the lowpass transforms, then
//! (for a digital filter) the bilinear transform with `fs = 2`.

use std::f64::consts::PI;

use nalgebra::DMatrix;
use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Session, Value};

use crate::numeric::from_dmatrix;
use crate::sigproc::{poly_from_roots, polynomial_roots};
use crate::util::{matrix_arg, need, positive, real_scalar, row};

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("buttord", buttord as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("cheb1ord", cheb1ord),
        ("cheb2ord", cheb2ord),
        ("ellipord", ellipord),
        ("ellip", ellip),
        ("besself", besself),
        ("iircomb", iircomb),
        ("tf2ss", tf2ss_fn),
        ("lp2lp", lp2lp_fn),
        ("lp2hp", lp2hp_fn),
        ("lp2bp", lp2bp_fn),
        ("lp2bs", lp2bs_fn),
    ] {
        session.register_builtin(name, func);
    }
}

struct Ss {
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    c: DMatrix<f64>,
    d: DMatrix<f64>,
}

fn value_matrix(m: &DMatrix<f64>) -> Value {
    Value::Matrix(from_dmatrix(m))
}

fn read_matrix(args: &[Value], index: usize) -> MatResult<DMatrix<f64>> {
    let m = matrix_arg(args, index)?;
    if !m.is_real() {
        return Err(MatError::new("state-space matrices must be real"));
    }
    Ok(DMatrix::from_fn(m.rows, m.cols, |r, c| m.get(r, c).re))
}

fn outputs4(ss: Ss) -> Value {
    Value::Multi(vec![
        value_matrix(&ss.a),
        value_matrix(&ss.b),
        value_matrix(&ss.c),
        value_matrix(&ss.d),
    ])
}

fn companion(num: &[f64], den: &[f64]) -> Ss {
    let mut den = den.to_vec();
    let mut num = num.to_vec();
    let scale = den[0];
    for c in den.iter_mut().chain(num.iter_mut()) {
        *c /= scale;
    }
    if num.len() < den.len() {
        let mut padded = vec![0.0; den.len() - num.len()];
        padded.extend(num);
        num = padded;
    }
    let n = den.len() - 1;
    let mut a = DMatrix::zeros(n, n);
    let mut b = DMatrix::zeros(n, 1);
    let mut c = DMatrix::zeros(1, n);
    if n > 0 {
        for col in 0..n {
            a[(0, col)] = -den[col + 1];
        }
        for row in 1..n {
            a[(row, row - 1)] = 1.0;
        }
        b[(0, 0)] = 1.0;
        for col in 0..n {
            c[(0, col)] = num[col + 1] - num[0] * den[col + 1];
        }
    }
    Ss { a, b, c, d: DMatrix::from_element(1, 1, num[0]) }
}

fn cascade(left: Ss, right: Ss) -> Ss {
    let n1 = left.a.nrows();
    let n2 = right.a.nrows();
    let mut a = DMatrix::zeros(n1 + n2, n1 + n2);
    if n1 > 0 {
        a.view_mut((0, 0), (n1, n1)).copy_from(&left.a);
    }
    if n2 > 0 {
        a.view_mut((n1, n1), (n2, n2)).copy_from(&right.a);
    }
    if n1 > 0 && n2 > 0 {
        let link = &right.b * &left.c;
        a.view_mut((n1, 0), (n2, n1)).copy_from(&link);
    }
    let b = if n1 == 0 {
        right.b.clone()
    } else if n2 == 0 {
        left.b.clone()
    } else {
        let lower = &right.b * &left.d;
        DMatrix::from_fn(n1 + n2, 1, |r, _| if r < n1 { left.b[(r, 0)] } else { lower[(r - n1, 0)] })
    };
    let c = if n2 == 0 {
        left.c.clone()
    } else if n1 == 0 {
        right.c.clone()
    } else {
        let left_part = &right.d * &left.c;
        DMatrix::from_fn(1, n1 + n2, |_, col| {
            if col < n1 { left_part[(0, col)] } else { right.c[(0, col - n1)] }
        })
    };
    Ss { a, b, c, d: &right.d * &left.d }
}

fn real_poly(roots: &[Complex64]) -> Vec<f64> {
    poly_from_roots(roots).iter().map(|c| c.re).collect()
}

/// Controllable canonical cascade. Real poles come first, then conjugate pairs, which is the
/// ordering behind `[A,B,C,D] = butter(3, 0.3)`.
fn zp2ss(zeros: &[Complex64], poles: &[Complex64], gain: f64) -> MatResult<Ss> {
    let (mut z_pairs, mut z_reals) = split_roots(zeros);
    let (p_pairs, p_reals) = split_roots(poles);
    let mut sections = Vec::new();
    for pole in p_reals {
        let num = if let Some(zero) = z_reals.pop() {
            vec![1.0, -zero]
        } else if let Some(pair) = z_pairs.pop() {
            // A complex pair cannot feed a single real pole; leave it for a later section.
            z_pairs.push(pair);
            vec![1.0]
        } else {
            vec![1.0]
        };
        sections.push(companion(&num, &[1.0, -pole]));
    }
    for pair in p_pairs {
        let den = real_poly(&[pair, pair.conj()]);
        let num = if let Some(zero) = z_pairs.pop() {
            real_poly(&[zero, zero.conj()])
        } else if z_reals.len() >= 2 {
            let z1 = z_reals.pop().unwrap();
            let z2 = z_reals.pop().unwrap();
            real_poly(&[Complex64::new(z1, 0.0), Complex64::new(z2, 0.0)])
        } else if let Some(zero) = z_reals.pop() {
            vec![1.0, -zero]
        } else {
            vec![1.0]
        };
        sections.push(companion(&num, &den));
    }
    if sections.is_empty() {
        return Ok(Ss {
            a: DMatrix::zeros(0, 0),
            b: DMatrix::zeros(0, 1),
            c: DMatrix::zeros(1, 0),
            d: DMatrix::from_element(1, 1, gain),
        });
    }
    let last = sections.len() - 1;
    sections[last].c *= gain;
    sections[last].d *= gain;
    let mut acc = sections.remove(0);
    for section in sections {
        acc = cascade(acc, section);
    }
    Ok(acc)
}

fn split_roots(roots: &[Complex64]) -> (Vec<Complex64>, Vec<f64>) {
    let mut pairs: Vec<Complex64> = roots
        .iter()
        .filter(|v| v.im.abs() > 1e-10 * (1.0 + v.re.abs()) && v.im > 0.0)
        .copied()
        .collect();
    pairs.sort_by(|a, b| a.re.total_cmp(&b.re).then(b.im.total_cmp(&a.im)));
    let mut reals: Vec<f64> = roots
        .iter()
        .filter(|v| v.im.abs() <= 1e-10 * (1.0 + v.re.abs()))
        .map(|v| v.re)
        .collect();
    reals.sort_by(|a, b| a.total_cmp(b));
    (pairs, reals)
}

fn lp2lp_ss(ss: Ss, wo: f64) -> Ss {
    Ss { a: ss.a * wo, b: ss.b * wo, c: ss.c, d: ss.d }
}

fn lp2hp_ss(ss: Ss, wo: f64) -> MatResult<Ss> {
    let inv = invert(&ss.a, "lp2hp")?;
    Ok(Ss {
        a: &inv * wo,
        b: &inv * &ss.b * -wo,
        c: &ss.c * &inv,
        d: &ss.d - &ss.c * &inv * &ss.b,
    })
}

fn lp2bp_ss(ss: Ss, wo: f64, bw: f64) -> Ss {
    let n = ss.a.nrows();
    let q = wo / bw;
    let mut a = DMatrix::zeros(2 * n, 2 * n);
    a.view_mut((0, 0), (n, n)).copy_from(&(&ss.a * q));
    a.view_mut((0, n), (n, n)).copy_from(&DMatrix::identity(n, n));
    a.view_mut((0, n), (n, n)).scale_mut(wo);
    a.view_mut((n, 0), (n, n)).copy_from(&DMatrix::identity(n, n));
    a.view_mut((n, 0), (n, n)).scale_mut(-wo);
    let mut b = DMatrix::zeros(2 * n, 1);
    b.view_mut((0, 0), (n, 1)).copy_from(&(&ss.b * q));
    let mut c = DMatrix::zeros(1, 2 * n);
    c.view_mut((0, 0), (1, n)).copy_from(&ss.c);
    Ss { a, b, c, d: ss.d }
}

fn lp2bs_ss(ss: Ss, wo: f64, bw: f64) -> MatResult<Ss> {
    let inv = invert(&ss.a, "lp2bs")?;
    let n = ss.a.nrows();
    let q = wo / bw;
    let mut a = DMatrix::zeros(2 * n, 2 * n);
    let top = &inv * q;
    a.view_mut((0, 0), (n, n)).copy_from(&top);
    a.view_mut((0, n), (n, n)).copy_from(&DMatrix::identity(n, n));
    a.view_mut((0, n), (n, n)).scale_mut(wo);
    a.view_mut((n, 0), (n, n)).copy_from(&DMatrix::identity(n, n));
    a.view_mut((n, 0), (n, n)).scale_mut(-wo);
    let mut b = DMatrix::zeros(2 * n, 1);
    let bin = &inv * &ss.b * -q;
    b.view_mut((0, 0), (n, 1)).copy_from(&bin);
    let mut c = DMatrix::zeros(1, 2 * n);
    let cin = &ss.c * &inv;
    c.view_mut((0, 0), (1, n)).copy_from(&cin);
    let d = &ss.d - &ss.c * &inv * &ss.b;
    Ok(Ss { a, b, c, d })
}

fn invert(a: &DMatrix<f64>, name: &str) -> MatResult<DMatrix<f64>> {
    a.clone()
        .try_inverse()
        .ok_or_else(|| MatError::new(format!("{name}: the state matrix is singular")))
}

/// Bilinear transform at `fs` samples per unit time. Normalized digital designs use `fs = 2`.
fn bilinear_ss(ss: Ss, fs: f64) -> MatResult<Ss> {
    let n = ss.a.nrows();
    if n == 0 {
        return Ok(ss);
    }
    let t = 1.0 / fs;
    let alpha = t / 2.0;
    let eye = DMatrix::<f64>::identity(n, n);
    let m = invert(&(eye.clone() - &ss.a * alpha), "bilinear")?;
    Ok(Ss {
        a: (eye + &ss.a * alpha) * &m,
        b: &m * &ss.b * t.sqrt(),
        c: &ss.c * &m * t.sqrt(),
        d: &ss.d + &ss.c * &m * &ss.b * alpha,
    })
}

/// State-space form of a lowpass prototype after `kind` (`low`, `high`, `pass`, `stop`).
/// `digital` applies the normalized bilinear transform (`fs = 2`).
pub(crate) fn realize_prototype(
    zeros: &[Complex64],
    poles: &[Complex64],
    gain: f64,
    kind: &str,
    center: f64,
    bandwidth: f64,
    digital: bool,
) -> MatResult<Value> {
    let ss = zp2ss(zeros, poles, gain)?;
    let ss = match kind {
        "high" => lp2hp_ss(ss, center)?,
        "pass" => lp2bp_ss(ss, center, bandwidth),
        "stop" => lp2bs_ss(ss, center, bandwidth)?,
        _ => lp2lp_ss(ss, center),
    };
    let ss = if digital { bilinear_ss(ss, 2.0)? } else { ss };
    Ok(outputs4(ss))
}

fn tf2ss_fn(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let b = matrix_arg(args, 0)?.reals()?;
    let a = matrix_arg(args, 1)?.reals()?;
    if a.iter().all(|c| *c == 0.0) {
        return Err(MatError::new("tf2ss denominator must not be zero"));
    }
    Ok(outputs4(companion(&b, &a)))
}

fn system(args: &[Value]) -> MatResult<Ss> {
    need(args, 4)?;
    Ok(Ss {
        a: read_matrix(args, 0)?,
        b: read_matrix(args, 1)?,
        c: read_matrix(args, 2)?,
        d: read_matrix(args, 3)?,
    })
}

fn lp2lp_fn(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 5)?;
    let wo = real_scalar(args, 4)?;
    Ok(outputs4(lp2lp_ss(system(args)?, wo)))
}

fn lp2hp_fn(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 5)?;
    let wo = real_scalar(args, 4)?;
    Ok(outputs4(lp2hp_ss(system(args)?, wo)?))
}

fn lp2bp_fn(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 6)?;
    Ok(outputs4(lp2bp_ss(system(args)?, real_scalar(args, 4)?, real_scalar(args, 5)?)))
}

fn lp2bs_fn(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 6)?;
    Ok(outputs4(lp2bs_ss(system(args)?, real_scalar(args, 4)?, real_scalar(args, 5)?)?))
}

// --- order selection -------------------------------------------------------

fn edges_of(value: &Value) -> MatResult<Vec<f64>> {
    let v = value.clone().into_matrix()?.reals()?;
    if v.is_empty() || v.len() > 2 {
        return Err(MatError::new("Wp and Ws need one or two frequencies"));
    }
    Ok(v)
}

fn prewarp(w: &[f64], analog: bool) -> Vec<f64> {
    if analog {
        w.to_vec()
    } else {
        w.iter().map(|v| (PI * v / 2.0).tan()).collect()
    }
}

fn postwarp(w: &[f64], analog: bool) -> Vec<f64> {
    if analog {
        w.to_vec()
    } else {
        w.iter().map(|v| v.atan() * 2.0 / PI).collect()
    }
}

/// Minimum transition ratio and the (possibly adjusted) passband edges, in the prewarped domain.
fn natural(stop: &[f64], pass: &[f64], gpass: f64, gstop: f64, kind: &str) -> (f64, Vec<f64>) {
    let mut pass = pass.to_vec();
    let nat = if pass.len() == 1 && stop.len() == 1 {
        if pass[0] < stop[0] {
            stop[0] / pass[0]
        } else {
            pass[0] / stop[0]
        }
    } else if pass[0] >= stop[0] {
        // Bandpass: the stopband edges sit outside the passband.
        let nat = (stop[1] * stop[1] - pass[0] * pass[1]) / (stop[1] * (pass[0] - pass[1]));
        let nat2 = (stop[0] * stop[0] - pass[0] * pass[1]) / (stop[0] * (pass[0] - pass[1]));
        nat.abs().min(nat2.abs())
    } else {
        // Bandstop: slide each pass edge toward the stopband until the order is smallest.
        let stop = stop.to_vec();
        let objective = |trial: &[f64]| {
            let values = [0, 1].map(|k| {
                (stop[k] * (trial[0] - trial[1])) / (stop[k] * stop[k] - trial[0] * trial[1])
            });
            order_real(values[0].abs().min(values[1].abs()), gpass, gstop, kind)
        };
        let lower = pass[0];
        let upper = pass[1];
        pass[0] = minimize(lower, stop[0] - 1e-12, |wp| trial_at(&[lower, upper], 0, wp, &objective));
        let moved = pass[0];
        pass[1] = minimize(stop[1] + 1e-12, upper, |wp| trial_at(&[moved, upper], 1, wp, &objective));
        let values = [0, 1].map(|k| {
            (stop[k] * (pass[0] - pass[1])) / (stop[k] * stop[k] - pass[0] * pass[1])
        });
        values[0].abs().min(values[1].abs())
    };
    (nat, pass)
}

fn trial_at(pass: &[f64], index: usize, wp: f64, objective: &impl Fn(&[f64]) -> f64) -> f64 {
    let mut trial = pass.to_vec();
    trial[index] = wp;
    objective(&trial)
}

fn minimize(mut left: f64, mut right: f64, f: impl Fn(f64) -> f64) -> f64 {
    let scale = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut c = right - (right - left) * scale;
    let mut d = left + (right - left) * scale;
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..80 {
        if fc < fd {
            right = d;
            d = c;
            fd = fc;
            c = right - (right - left) * scale;
            fc = f(c);
        } else {
            left = c;
            c = d;
            fc = fd;
            d = left + (right - left) * scale;
            fd = f(d);
        }
    }
    (left + right) / 2.0
}

fn order_real(nat: f64, gpass: f64, gstop: f64, kind: &str) -> f64 {
    let gpass = 10f64.powf(0.1 * gpass);
    let gstop = 10f64.powf(0.1 * gstop);
    match kind {
        "butter" => ((gstop - 1.0) / (gpass - 1.0)).log10() / (2.0 * nat.log10()),
        "ellip" => {
            let arg1 = (gpass - 1.0) / (gstop - 1.0);
            let arg0 = 1.0 / nat;
            let (k0, k0p) = (ellipk(arg0 * arg0), ellipk(1.0 - arg0 * arg0));
            let (k1, k1p) = (ellipk(arg1), ellipk(1.0 - arg1));
            k0 * k1p / (k0p * k1)
        }
        _ => ((gstop - 1.0) / (gpass - 1.0)).sqrt().acosh() / nat.acosh(),
    }
}

fn order_result(n: f64, wn: Vec<f64>) -> Value {
    let wn = if wn.len() == 1 {
        Value::scalar(wn[0])
    } else {
        crate::util::col(wn)
    };
    Value::Multi(vec![Value::scalar(n.ceil()), wn])
}

fn spec(args: &[Value], name: &str) -> MatResult<(Vec<f64>, Vec<f64>, f64, f64, bool)> {
    need(args, 4)?;
    let wp = edges_of(&args[0])?;
    let ws = edges_of(&args[1])?;
    if wp.len() != ws.len() {
        return Err(MatError::new(format!("{name}: Wp and Ws must have the same length")));
    }
    let gpass = real_scalar(args, 2)?;
    let gstop = real_scalar(args, 3)?;
    let mut analog = false;
    for arg in &args[4..] {
        match arg {
            Value::Char(text) if text == "s" => analog = true,
            Value::Char(text) if text == "z" => analog = false,
            Value::Char(text) => {
                return Err(MatError::new(format!("{name} does not know the option '{text}'")));
            }
            _ => return Err(MatError::new(format!("{name} options must be strings"))),
        }
    }
    Ok((wp, ws, gpass, gstop, analog))
}

fn buttord(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    // `Wn` is the cutoff that meets the stopband exactly after the order is rounded up
    // (MATLAB R2025a). Lowpass and highpass follow that directly. A band filter is made
    // symmetric about the passband (bandpass) or stopband (bandstop) center first.
    let (wp, ws, gpass, gstop, analog) = spec(args, "buttord")?;
    let mut pass = prewarp(&wp, analog);
    let mut stop = prewarp(&ws, analog);
    let (nat, bandpass) = if pass.len() == 2 {
        if pass[0] > stop[0] {
            if pass[0] * pass[1] < stop[0] * stop[1] {
                stop[1] = pass[0] * pass[1] / stop[0];
            } else {
                stop[0] = pass[0] * pass[1] / stop[1];
            }
            ((stop[1] - stop[0]) / (pass[1] - pass[0]), true)
        } else {
            if pass[0] * pass[1] > stop[0] * stop[1] {
                pass[1] = stop[0] * stop[1] / pass[0];
            } else {
                pass[0] = stop[0] * stop[1] / pass[1];
            }
            let center = pass[0] * pass[1];
            ((center / (stop[1] - stop[0])) / (center / (pass[1] - pass[0])), false)
        }
    } else if pass[0] > stop[0] {
        (pass[0] / stop[0], false)
    } else {
        (stop[0] / pass[0], false)
    };
    let qs = (10f64.powf(gstop / 10.0) - 1.0).ln();
    let qp = (10f64.powf(gpass / 10.0) - 1.0).ln();
    let n = (0.5 * (qs - qp) / nat.ln()).ceil();
    let wn = if pass.len() == 2 {
        let (edges, sign) = if bandpass { (&stop, -1.0) } else { (&stop, 1.0) };
        let shifted = edges
            .iter()
            .map(|w| (w.ln() + sign * qs / (2.0 * n)).exp())
            .collect::<Vec<_>>();
        let w0 = (edges[0] * edges[1]).sqrt();
        let q = w0 / (edges[1] - edges[0]);
        let ratio = shifted[0] / edges[0];
        let root = (ratio * ratio + 4.0 * q * q).sqrt();
        let mut pair = [(ratio + root).abs(), (ratio - root).abs()]
            .map(|w| w / (2.0 * q / w0));
        pair.sort_by(|a, b| a.total_cmp(b));
        pair.to_vec()
    } else if wp[0] > ws[0] {
        vec![(stop[0].ln() + qs / (2.0 * n)).exp()]
    } else {
        vec![(stop[0].ln() - qs / (2.0 * n)).exp()]
    };
    Ok(order_result(n, postwarp(&wn, analog)))
}

fn cheb1ord(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (wp, ws, gpass, gstop, analog) = spec(args, "cheb1ord")?;
    let (nat, _) = natural(&prewarp(&ws, analog), &prewarp(&wp, analog), gpass, gstop, "cheby");
    Ok(order_result(order_real(nat, gpass, gstop, "cheby"), wp))
}

fn cheb2ord(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    // MATLAB R2025a returns the stopband edge, not the 3 dB frequency.
    let (wp, ws, gpass, gstop, analog) = spec(args, "cheb2ord")?;
    let (nat, _) = natural(&prewarp(&ws, analog), &prewarp(&wp, analog), gpass, gstop, "cheby");
    Ok(order_result(order_real(nat, gpass, gstop, "cheby"), ws))
}

fn ellipord(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (wp, ws, gpass, gstop, analog) = spec(args, "ellipord")?;
    let (nat, _) = natural(&prewarp(&ws, analog), &prewarp(&wp, analog), gpass, gstop, "ellip");
    Ok(order_result(order_real(nat, gpass, gstop, "ellip"), wp))
}

// --- elliptic integrals and the elliptic prototype -------------------------

fn ellipk(m: f64) -> f64 {
    if m <= 0.0 {
        return PI / 2.0;
    }
    if m >= 1.0 {
        return f64::INFINITY;
    }
    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();
    for _ in 0..40 {
        let next_a = (a + b) / 2.0;
        let next_b = (a * b).sqrt();
        if (a - b).abs() <= 1e-15 * a {
            a = next_a;
            break;
        }
        a = next_a;
        b = next_b;
    }
    PI / (2.0 * a)
}

/// Jacobi sn, cn, dn. Arithmetic-geometric mean, as in Cephes `ellpj`.
fn ellipj(u: f64, m: f64) -> (f64, f64, f64) {
    if m <= 0.0 {
        return (u.sin(), u.cos(), 1.0);
    }
    if m >= 1.0 {
        let cn = 1.0 / u.cosh();
        return (u.tanh(), cn, cn);
    }
    let mut a = [0.0; 10];
    let mut c = [0.0; 10];
    a[0] = 1.0;
    let mut b = (1.0 - m).sqrt();
    c[0] = m.sqrt();
    let mut twon = 1.0;
    let mut i = 0;
    while (c[i] / a[i]).abs() > 1e-16 && i < 8 {
        let ai = a[i];
        i += 1;
        c[i] = (ai - b) / 2.0;
        let t = (ai * b).sqrt();
        a[i] = (ai + b) / 2.0;
        b = t;
        twon *= 2.0;
    }
    let mut phi = twon * a[i] * u;
    let mut previous = phi;
    while i > 0 {
        let t = c[i] * phi.sin() / a[i];
        previous = phi;
        phi = (t.asin() + phi) / 2.0;
        i -= 1;
    }
    let sn = phi.sin();
    let cn = phi.cos();
    let dn = cn / (phi - previous).cos();
    (sn, cn, dn)
}

fn ellip_modulus(n: usize, m1: f64) -> f64 {
    let q1 = (-PI * ellipk(1.0 - m1) / ellipk(m1)).exp();
    let q = q1.powf(1.0 / n as f64);
    let mut num = 0.0;
    let mut den = 0.0;
    for k in 0..=7 {
        num += q.powi((k * (k + 1)) as i32);
    }
    for k in 1..=8 {
        den += q.powi((k * k) as i32);
    }
    16.0 * q * (num / (1.0 + 2.0 * den)).powi(4)
}

fn arc_jac_sc1(w: f64, m: f64) -> f64 {
    // sc(z, 1-m) = -i sn(i z, m), so the inverse of a real `w` is imaginary.
    let z = arc_jac_sn(Complex64::new(0.0, w), m);
    z.im
}

fn arc_jac_sn(w: Complex64, m: f64) -> Complex64 {
    let complement = |z: Complex64| (Complex64::new(1.0, 0.0) - z * z).sqrt();
    let k = m.sqrt();
    if k <= 0.0 {
        return w.asin();
    }
    if k >= 1.0 {
        return w.atanh();
    }
    let mut ks = vec![k];
    for _ in 0..12 {
        let last = *ks.last().unwrap();
        if last.abs() < 1e-16 {
            break;
        }
        let kp = (1.0 - last * last).sqrt();
        ks.push((1.0 - kp) / (1.0 + kp));
    }
    let mut big_k = PI / 2.0;
    for &kn in ks.iter().skip(1) {
        big_k *= 1.0 + kn;
    }
    let mut wn = w;
    for pair in ks.windows(2) {
        let knext = pair[1];
        wn = (wn * 2.0) / ((1.0 + knext) * (Complex64::new(1.0, 0.0) + complement(wn * pair[0])));
    }
    wn.asin() * (2.0 / PI) * big_k
}

/// Analog lowpass elliptic prototype with cutoff 1 and ripples `rp`, `rs` in dB.
pub(crate) fn elliptic_prototype(n: usize, rp: f64, rs: f64) -> (Vec<Complex64>, Vec<Complex64>, f64) {
    if n == 1 {
        let p = -(1.0 / (10f64.powf(0.1 * rp) - 1.0)).sqrt();
        return (Vec::new(), vec![Complex64::new(p, 0.0)], -p);
    }
    let eps_sq = 10f64.powf(0.1 * rp) - 1.0;
    let eps = eps_sq.sqrt();
    let ck1_sq = eps_sq / (10f64.powf(0.1 * rs) - 1.0);
    let m = ellip_modulus(n, ck1_sq);
    let capk = ellipk(m);
    let indices: Vec<f64> = (0..n).map(|k| (1 - (n % 2) + 2 * k) as f64).filter(|j| *j < n as f64).collect();
    let mut s = Vec::new();
    let mut c = Vec::new();
    let mut d = Vec::new();
    for j in &indices {
        let (sj, cj, dj) = ellipj(j * capk / n as f64, m);
        s.push(sj);
        c.push(cj);
        d.push(dj);
    }
    let mut zeros = Vec::new();
    for &sj in &s {
        if sj.abs() > 1e-14 {
            let z = Complex64::new(0.0, 1.0) / (m.sqrt() * sj);
            zeros.push(z);
            zeros.push(z.conj());
        }
    }
    let r = arc_jac_sc1(1.0 / eps, ck1_sq);
    let v0 = capk * r / (n as f64 * ellipk(ck1_sq));
    let (sv, cv, dv) = ellipj(v0, 1.0 - m);
    let mut poles = Vec::new();
    for i in 0..s.len() {
        let denom = 1.0 - (d[i] * sv).powi(2);
        let p = -Complex64::new(c[i] * d[i] * sv * cv, s[i] * dv) / denom;
        if n % 2 == 1 && p.im.abs() <= 1e-10 {
            poles.push(Complex64::new(p.re, 0.0));
        } else {
            poles.push(p);
            poles.push(p.conj());
        }
    }
    let prod = |v: &[Complex64]| v.iter().fold(Complex64::new(1.0, 0.0), |acc, z| acc * -z);
    let mut gain = (prod(&poles) / prod(&zeros)).re;
    if n % 2 == 0 {
        gain /= (1.0 + eps_sq).sqrt();
    }
    (zeros, poles, gain)
}

fn bessel_denominator(n: usize) -> Vec<f64> {
    let falling = |mut x: usize, count: usize| -> f64 {
        let mut v = 1.0;
        for _ in 0..count {
            v *= x as f64;
            x -= 1;
        }
        v
    };
    let mut fact = 1.0;
    let mut low = Vec::with_capacity(n + 1);
    for k in 0..=n {
        if k > 0 {
            fact *= k as f64;
        }
        low.push(falling(2 * n - k, n) / (2f64.powi((n - k) as i32) * fact));
    }
    low.reverse();
    low
}

/// Phase-normalized analog Bessel lowpass prototype (MATLAB's `besself` normalization).
pub(crate) fn bessel_prototype(n: usize) -> MatResult<(Vec<Complex64>, Vec<Complex64>, f64)> {
    let den = bessel_denominator(n);
    let a_last = den[den.len() - 1];
    let mut poles = polynomial_roots(&den)?;
    let scale = a_last.powf(-1.0 / n as f64);
    for pole in &mut poles {
        *pole *= scale;
    }
    Ok((Vec::new(), poles, 1.0))
}

fn ellip(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    crate::iir::design_prototype(session, args, "ellip", 2, false, |n, rp, rs| {
        Ok(elliptic_prototype(n, rp, rs))
    })
}

fn besself(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    crate::iir::design_prototype(session, args, "besself", 0, true, |n, _, _| bessel_prototype(n))
}

fn iircomb(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let n = positive(real_scalar(args, 0)?)?;
    if n == 0 {
        return Err(MatError::new("iircomb order must be at least 1"));
    }
    let bw = real_scalar(args, 1)?;
    if !(bw > 0.0 && bw < 1.0) {
        return Err(MatError::new("iircomb: bw must be between 0 and 1"));
    }
    let mut peak = false;
    let mut ab = None;
    for arg in &args[2..] {
        match arg {
            Value::Char(text) if text.eq_ignore_ascii_case("notch") => peak = false,
            Value::Char(text) if text.eq_ignore_ascii_case("peak") => peak = true,
            Value::Char(text) => {
                return Err(MatError::new(format!("iircomb does not know the option '{text}'")));
            }
            other => ab = Some(other.real_scalar()?),
        }
    }
    let ratio = if let Some(ab) = ab {
        if !ab.is_finite() || ab <= 0.0 {
            return Err(MatError::new("iircomb: Ab must be positive"));
        }
        (10f64.powf(ab / 10.0) - 1.0).sqrt()
    } else {
        1.0
    };
    let ratio = if peak { 1.0 / ratio } else { ratio };
    let beta = ratio * (n as f64 * bw * PI / 4.0).tan();
    let b0 = if peak { beta / (1.0 + beta) } else { 1.0 / (1.0 + beta) };
    let a_last = (1.0 - beta) / (1.0 + beta);
    let mut b = vec![0.0; n + 1];
    let mut a = vec![0.0; n + 1];
    b[0] = b0;
    b[n] = -b0;
    a[0] = 1.0;
    a[n] = if peak { a_last } else { -a_last };
    Ok(if session.nargout >= 2 {
        Value::Multi(vec![row(b), row(a)])
    } else {
        row(b)
    })
}

