use std::f64::consts::PI;

use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Session, Value};

use crate::filt::complex_column;
use crate::sigproc::{poly_from_roots, polynomial_roots};
use crate::util::{matrix_arg, need, positive, real_scalar, row};

/// Zeros, poles, and gain.
type Zpk = (Vec<Complex64>, Vec<Complex64>, f64);

/// The sampling frequency of the bilinear transform for frequencies normalized to Nyquist.
const NORMALIZED_FS2: f64 = 4.0;

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("butter", butter as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("cheby1", cheby1),
        ("cheby2", cheby2),
        ("bilinear", bilinear),
        ("iirnotch", iirnotch),
        ("iirpeak", iirpeak),
    ] {
        session.register_builtin(name, func);
    }
}

fn product_of_negated(values: &[Complex64]) -> Complex64 {
    values.iter().fold(Complex64::new(1.0, 0.0), |acc, v| acc * -v)
}

// Analog lowpass prototypes with a cutoff of 1 rad/s

fn butterworth_prototype(n: usize) -> Zpk {
    let poles = (0..n)
        .map(|k| Complex64::from_polar(1.0, PI * (2 * k + n + 1) as f64 / (2 * n) as f64))
        .collect();
    (Vec::new(), poles, 1.0)
}

/// Odd indices `-(n - 1), -(n - 3), ..., n - 1`.
fn symmetric_steps(n: usize) -> impl Iterator<Item = f64> {
    (-(n as i64) + 1..n as i64).step_by(2).map(|m| m as f64)
}

fn chebyshev1_prototype(n: usize, ripple_db: f64) -> Zpk {
    let eps = (10f64.powf(0.1 * ripple_db) - 1.0).sqrt();
    let mu = (1.0 / eps).asinh() / n as f64;
    let poles: Vec<Complex64> = symmetric_steps(n)
        .map(|m| -Complex64::new(mu, PI * m / (2.0 * n as f64)).sinh())
        .collect();
    let mut gain = product_of_negated(&poles).re;
    if n.is_multiple_of(2) {
        gain /= (1.0 + eps * eps).sqrt();
    }
    (Vec::new(), poles, gain)
}

fn chebyshev2_prototype(n: usize, stopband_db: f64) -> Zpk {
    let delta = 1.0 / (10f64.powf(0.1 * stopband_db) - 1.0).sqrt();
    let mu = (1.0 / delta).asinh() / n as f64;
    // A zero at infinity (cos = 0, odd order) is left out.
    let zeros: Vec<Complex64> = (1..=n)
        .map(|k| (2 * k - 1) as f64 * PI / (2 * n) as f64)
        .map(f64::cos)
        .filter(|c| c.abs() > 1e-12)
        .map(|c| Complex64::new(0.0, 1.0) / c)
        .collect();
    let poles: Vec<Complex64> = symmetric_steps(n)
        .map(|m| {
            let base = -Complex64::from_polar(1.0, PI * m / (2.0 * n as f64));
            let warped = Complex64::new(mu.sinh() * base.re, mu.cosh() * base.im);
            1.0 / warped
        })
        .collect();
    let gain = (product_of_negated(&poles) / product_of_negated(&zeros)).re;
    (zeros, poles, gain)
}

// Frequency transformations of a lowpass prototype in zpk form

fn degree(z: &[Complex64], p: &[Complex64]) -> usize {
    p.len().saturating_sub(z.len())
}

fn lp2lp((z, p, k): Zpk, wo: f64) -> Zpk {
    let d = degree(&z, &p) as i32;
    (
        z.iter().map(|v| v * wo).collect(),
        p.iter().map(|v| v * wo).collect(),
        k * wo.powi(d),
    )
}

fn lp2hp((z, p, k): Zpk, wo: f64) -> Zpk {
    let d = degree(&z, &p);
    let mut zeros: Vec<Complex64> = z.iter().map(|v| wo / v).collect();
    zeros.extend(std::iter::repeat_n(Complex64::new(0.0, 0.0), d));
    let poles = p.iter().map(|v| wo / v).collect();
    let gain = k * (product_of_negated(&z) / product_of_negated(&p)).re;
    (zeros, poles, gain)
}

/// Each root `r` becomes the pair `r +- sqrt(r^2 - wo^2)`, all sums first and then all differences.
fn split_pairs(roots: &[Complex64], wo: f64) -> Vec<Complex64> {
    let root = |r: &Complex64| (r * r - wo * wo).sqrt();
    roots
        .iter()
        .map(|r| r + root(r))
        .chain(roots.iter().map(|r| r - root(r)))
        .collect()
}

fn lp2bp((z, p, k): Zpk, wo: f64, bw: f64) -> Zpk {
    let d = degree(&z, &p);
    let half = |v: &[Complex64]| -> Vec<Complex64> { v.iter().map(|r| r * bw / 2.0).collect() };
    let mut zeros = split_pairs(&half(&z), wo);
    zeros.extend(std::iter::repeat_n(Complex64::new(0.0, 0.0), d));
    (zeros, split_pairs(&half(&p), wo), k * bw.powi(d as i32))
}

fn lp2bs((z, p, k): Zpk, wo: f64, bw: f64) -> Zpk {
    let d = degree(&z, &p);
    let invert = |v: &[Complex64]| -> Vec<Complex64> { v.iter().map(|r| (bw / 2.0) / r).collect() };
    let mut zeros = split_pairs(&invert(&z), wo);
    zeros.extend(std::iter::repeat_n(Complex64::new(0.0, wo), d));
    zeros.extend(std::iter::repeat_n(Complex64::new(0.0, -wo), d));
    let gain = k * (product_of_negated(&z) / product_of_negated(&p)).re;
    (zeros, split_pairs(&invert(&p), wo), gain)
}

/// Bilinear transform with `s = fs2 * (z - 1) / (z + 1)`. Missing zeros are placed at -1.
fn bilinear_zpk((z, p, k): Zpk, fs2: f64) -> Zpk {
    let d = degree(&z, &p);
    let map = |r: &Complex64| (fs2 + r) / (fs2 - r);
    let mut zeros: Vec<Complex64> = z.iter().map(map).collect();
    zeros.extend(std::iter::repeat_n(Complex64::new(-1.0, 0.0), d));
    let poles = p.iter().map(map).collect();
    let shifted = |v: &[Complex64]| -> Complex64 {
        v.iter().fold(Complex64::new(1.0, 0.0), |acc, r| acc * (fs2 - r))
    };
    (zeros, poles, k * (shifted(&z) / shifted(&p)).re)
}

fn finish(session: &Session, (z, p, k): Zpk) -> Value {
    if session.nargout >= 3 {
        return Value::Multi(vec![complex_column(z), complex_column(p), Value::scalar(k)]);
    }
    let b: Vec<f64> = poly_from_roots(&z).iter().map(|c| c.re * k).collect();
    let a: Vec<f64> = poly_from_roots(&p).iter().map(|c| c.re).collect();
    if session.nargout >= 2 {
        Value::Multi(vec![row(b), row(a)])
    } else {
        row(b)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Band {
    Low,
    High,
    Pass,
    Stop,
}

/// Shared argument handling of `butter`, `cheby1`, and `cheby2`.
/// `extra` is 1 when the prototype takes a ripple argument before `Wn`.
fn design(
    session: &mut Session,
    args: &[Value],
    name: &str,
    extra: usize,
    prototype: impl Fn(usize, f64) -> Zpk,
) -> MatResult<Value> {
    need(args, 2 + extra)?;
    let n = positive(real_scalar(args, 0)?)?;
    if n == 0 {
        return Err(MatError::new(format!("{name} order must be at least 1")));
    }
    let ripple = if extra == 1 { real_scalar(args, 1)? } else { 0.0 };
    if extra == 1 && ripple <= 0.0 {
        return Err(MatError::new(format!("{name} ripple in dB must be positive")));
    }
    let wn = matrix_arg(args, 1 + extra)?.reals()?;
    let mut band = None;
    let mut analog = false;
    for arg in &args[2 + extra..] {
        let Value::Char(option) = arg else {
            return Err(MatError::new(format!("{name} options must be strings")));
        };
        match option.as_str() {
            "low" | "lowpass" => band = Some(Band::Low),
            "high" | "highpass" => band = Some(Band::High),
            "bandpass" => band = Some(Band::Pass),
            "stop" | "bandstop" => band = Some(Band::Stop),
            "s" => analog = true,
            "z" => analog = false,
            other => return Err(MatError::new(format!("{name} does not know the option '{other}'"))),
        }
    }
    let band = band.unwrap_or(if wn.len() == 2 { Band::Pass } else { Band::Low });
    let expected = if matches!(band, Band::Pass | Band::Stop) { 2 } else { 1 };
    if wn.len() != expected {
        return Err(MatError::new(format!(
            "{name}: Wn needs {expected} element(s) for this filter type"
        )));
    }
    if expected == 2 && wn[0] >= wn[1] {
        return Err(MatError::new(format!("{name}: Wn(1) must be smaller than Wn(2)")));
    }
    let valid = |w: &f64| if analog { *w > 0.0 } else { *w > 0.0 && *w < 1.0 };
    if !wn.iter().all(valid) {
        return Err(MatError::new(if analog {
            format!("{name}: Wn must be positive")
        } else {
            format!("{name}: Wn must be between 0 and 1")
        }));
    }
    let edges: Vec<f64> = if analog {
        wn
    } else {
        wn.iter()
            .map(|w| NORMALIZED_FS2 * (PI * w / 2.0).tan())
            .collect()
    };
    let lowpass = prototype(n, ripple);
    let transformed = match band {
        Band::Low => lp2lp(lowpass, edges[0]),
        Band::High => lp2hp(lowpass, edges[0]),
        Band::Pass => lp2bp(lowpass, (edges[0] * edges[1]).sqrt(), edges[1] - edges[0]),
        Band::Stop => lp2bs(lowpass, (edges[0] * edges[1]).sqrt(), edges[1] - edges[0]),
    };
    let result = if analog {
        transformed
    } else {
        bilinear_zpk(transformed, NORMALIZED_FS2)
    };
    Ok(finish(session, result))
}

fn butter(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    design(session, args, "butter", 0, |n, _| butterworth_prototype(n))
}

fn cheby1(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    design(session, args, "cheby1", 1, chebyshev1_prototype)
}

fn cheby2(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    design(session, args, "cheby2", 1, chebyshev2_prototype)
}

fn bilinear(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    if args.len() >= 4 {
        let z = matrix_arg(args, 0)?.data;
        let p = matrix_arg(args, 1)?.data;
        if z.len() > p.len() {
            return Err(MatError::new("bilinear needs a proper transfer function"));
        }
        let fs = real_scalar(args, 3)?;
        let (zd, pd, kd) = bilinear_zpk((z, p, real_scalar(args, 2)?), 2.0 * fs);
        return Ok(if session.nargout >= 3 {
            Value::Multi(vec![complex_column(zd), complex_column(pd), Value::scalar(kd)])
        } else {
            complex_column(zd)
        });
    }
    let num = matrix_arg(args, 0)?.reals()?;
    let den = matrix_arg(args, 1)?.reals()?;
    let first = |p: &[f64]| p.iter().copied().find(|c| *c != 0.0);
    let (Some(num0), Some(den0)) = (first(&num), first(&den)) else {
        return Err(MatError::new("bilinear needs a non-zero numerator and denominator"));
    };
    let z = polynomial_roots(&num)?;
    let p = polynomial_roots(&den)?;
    if z.len() > p.len() {
        return Err(MatError::new("bilinear needs a proper transfer function"));
    }
    let fs = real_scalar(args, 2)?;
    let (zd, pd, kd) = bilinear_zpk((z, p, num0 / den0), 2.0 * fs);
    Ok(finish_tf(session, (zd, pd, kd)))
}

fn finish_tf(session: &Session, (z, p, k): Zpk) -> Value {
    let b: Vec<f64> = poly_from_roots(&z).iter().map(|c| c.re * k).collect();
    let a: Vec<f64> = poly_from_roots(&p).iter().map(|c| c.re).collect();
    if session.nargout >= 2 {
        Value::Multi(vec![row(b), row(a)])
    } else {
        row(b)
    }
}

/// Second-order notch or peak filter at `w0` with a -3 dB bandwidth `bw`, both normalized to Nyquist.
fn resonator(session: &mut Session, args: &[Value], name: &str, peak: bool) -> MatResult<Value> {
    need(args, 2)?;
    if args.len() > 2 {
        return Err(MatError::new(format!("{name} with a bandwidth gain is not supported")));
    }
    let w0 = real_scalar(args, 0)?;
    let bw = real_scalar(args, 1)?;
    if !(w0 > 0.0 && w0 < 1.0) || !(bw > 0.0 && bw < 1.0) {
        return Err(MatError::new(format!("{name}: w0 and bw must be between 0 and 1")));
    }
    let beta = (bw * PI / 2.0).tan();
    let cosine = (PI * w0).cos();
    let scale = 1.0 + beta;
    let a = vec![1.0, -2.0 * cosine / scale, (1.0 - beta) / scale];
    let b = if peak {
        vec![beta / scale, 0.0, -beta / scale]
    } else {
        vec![1.0 / scale, -2.0 * cosine / scale, 1.0 / scale]
    };
    Ok(if session.nargout >= 2 {
        Value::Multi(vec![row(b), row(a)])
    } else {
        row(b)
    })
}

fn iirnotch(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    resonator(session, args, "iirnotch", false)
}

fn iirpeak(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    resonator(session, args, "iirpeak", true)
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

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

    /// Magnitude of `b(z) / a(z)` at the normalized frequency `w` (1 is Nyquist).
    fn gain(b: &[f64], a: &[f64], w: f64) -> f64 {
        let z = Complex64::from_polar(1.0, -std::f64::consts::PI * w);
        let eval = |c: &[f64]| {
            c.iter()
                .enumerate()
                .map(|(k, v)| z.powi(k as i32) * v)
                .sum::<Complex64>()
        };
        (eval(b) / eval(a)).norm()
    }

    fn design(src: &str) -> (Vec<f64>, Vec<f64>) {
        (reals(src, "b"), reals(src, "a"))
    }

    const HALF_POWER: f64 = std::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn butterworth_matches_known_coefficients() {
        let (b, a) = design("[b, a] = butter(2, 0.2);");
        assert_close(&b, &[0.0674552738890719, 0.134910547778144, 0.0674552738890719], 1e-9);
        assert_close(&a, &[1.0, -1.14298050253990, 0.412801598096189], 1e-9);
        let (b, a) = design("[b, a] = butter(1, 0.5);");
        assert_close(&b, &[0.5, 0.5], 1e-12);
        assert_close(&a, &[1.0, 0.0], 1e-12);
    }

    #[test]
    fn butterworth_has_half_power_at_the_cutoff_for_every_type() {
        let (b, a) = design("[b, a] = butter(4, 0.3);");
        assert!((gain(&b, &a, 0.3) - HALF_POWER).abs() < 1e-9);
        assert!((gain(&b, &a, 0.0) - 1.0).abs() < 1e-9);
        let (b, a) = design("[b, a] = butter(3, 0.4, 'high');");
        assert!((gain(&b, &a, 0.4) - HALF_POWER).abs() < 1e-9);
        assert!(gain(&b, &a, 0.0) < 1e-9 && (gain(&b, &a, 1.0) - 1.0).abs() < 1e-9);
        let (b, a) = design("[b, a] = butter(3, [0.2 0.4]);");
        assert_eq!((b.len(), a.len()), (7, 7));
        assert!((gain(&b, &a, 0.2) - HALF_POWER).abs() < 1e-9);
        assert!((gain(&b, &a, 0.4) - HALF_POWER).abs() < 1e-9);
        assert!(gain(&b, &a, 0.0) < 1e-9 && gain(&b, &a, 1.0) < 1e-9);
        let (b, a) = design("[b, a] = butter(2, [0.2 0.4], 'stop');");
        assert_eq!((b.len(), a.len()), (5, 5));
        assert!((gain(&b, &a, 0.2) - HALF_POWER).abs() < 1e-9);
        assert!((gain(&b, &a, 0.4) - HALF_POWER).abs() < 1e-9);
        assert!((gain(&b, &a, 0.0) - 1.0).abs() < 1e-9 && (gain(&b, &a, 1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn chebyshev_filters_meet_their_ripple_and_attenuation() {
        let ripple = 10f64.powf(-1.0 / 20.0);
        let (b, a) = design("[b, a] = cheby1(3, 1, 0.4);");
        assert!((gain(&b, &a, 0.0) - 1.0).abs() < 1e-9);
        assert!((gain(&b, &a, 0.4) - ripple).abs() < 1e-9);
        let (b, a) = design("[b, a] = cheby1(4, 1, 0.4);");
        assert!((gain(&b, &a, 0.0) - ripple).abs() < 1e-9);
        assert!((gain(&b, &a, 0.4) - ripple).abs() < 1e-9);
        let (b, a) = design("[b, a] = cheby1(3, 1, 0.4, 'high');");
        assert!((gain(&b, &a, 0.4) - ripple).abs() < 1e-9 && gain(&b, &a, 0.0) < 1e-9);
        let (b, a) = design("[b, a] = cheby2(4, 40, 0.5);");
        assert!((gain(&b, &a, 0.5) - 0.01).abs() < 1e-9);
        assert!((gain(&b, &a, 0.0) - 1.0).abs() < 1e-9);
        let (b, a) = design("[b, a] = cheby2(5, 30, 0.4);");
        assert!((gain(&b, &a, 0.4) - 10f64.powf(-30.0 / 20.0)).abs() < 1e-9);
        let (b, a) = design("[b, a] = cheby1(2, 1, [0.2 0.4]);");
        assert_eq!(b.len(), 5);
        assert!((gain(&b, &a, 0.2) - ripple).abs() < 1e-9 && (gain(&b, &a, 0.4) - ripple).abs() < 1e-9);
        let (b, a) = design("[b, a] = cheby2(2, 30, [0.2 0.4], 'stop');");
        assert!((gain(&b, &a, 0.0) - 1.0).abs() < 1e-9 && (gain(&b, &a, 1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn zero_pole_gain_output_and_stability() {
        let session = exec("[z, p, k] = butter(3, 0.3); r = max(abs(p)); nz = numel(z);").unwrap();
        assert!(session.get("r").unwrap().real_scalar().unwrap() < 1.0);
        assert_eq!(session.get("nz").unwrap().real_scalar().unwrap(), 3.0);
        let session = exec("[z, p, k] = cheby2(4, 40, 0.5); r = max(abs(p)); nz = numel(z);").unwrap();
        assert!(session.get("r").unwrap().real_scalar().unwrap() < 1.0);
        assert_eq!(session.get("nz").unwrap().real_scalar().unwrap(), 4.0);
    }

    #[test]
    fn analog_designs() {
        let (b, a) = design("[b, a] = butter(2, 1, 's');");
        assert_close(&b, &[1.0], 1e-12);
        assert_close(&a, &[1.0, std::f64::consts::SQRT_2, 1.0], 1e-12);
        let (b, a) = design("[b, a] = butter(2, 1, 'high', 's');");
        assert_close(&b, &[1.0, 0.0, 0.0], 1e-12);
        assert_close(&a, &[1.0, std::f64::consts::SQRT_2, 1.0], 1e-12);
        let (b, a) = design("[b, a] = butter(2, 10, 's');");
        assert_close(&b, &[100.0], 1e-9);
        assert_close(&a, &[1.0, 10.0 * std::f64::consts::SQRT_2, 100.0], 1e-9);
    }

    #[test]
    fn bilinear_transform() {
        let (b, a) = design("[b, a] = bilinear(1, [1 1], 1);");
        assert_close(&b, &[1.0 / 3.0, 1.0 / 3.0], 1e-12);
        assert_close(&a, &[1.0, -1.0 / 3.0], 1e-12);
        let session = exec("[zd, pd, kd] = bilinear([], [-1], 1, 1);").unwrap();
        assert!((session.get("kd").unwrap().real_scalar().unwrap() - 1.0 / 3.0).abs() < 1e-12);
        assert!(crate::exec("[b, a] = bilinear([1 1 1], [1 1], 1);").is_err());
    }

    #[test]
    fn notch_and_peak_filters() {
        let (b, a) = design("[b, a] = iirnotch(0.5, 0.1);");
        assert!(gain(&b, &a, 0.5) < 1e-12);
        assert!((gain(&b, &a, 0.0) - 1.0).abs() < 1e-12 && (gain(&b, &a, 1.0) - 1.0).abs() < 1e-12);
        let half_width = 0.05;
        assert!((gain(&b, &a, 0.5 + half_width) - HALF_POWER).abs() < 0.02);
        let (b, a) = design("[b, a] = iirpeak(0.5, 0.1);");
        assert!((gain(&b, &a, 0.5) - 1.0).abs() < 1e-12);
        assert!(gain(&b, &a, 0.0) < 1e-12 && gain(&b, &a, 1.0) < 1e-12);
        assert!(crate::exec("[b, a] = iirnotch(0.5, 0.1, 6);").is_err());
    }

    #[test]
    fn design_arguments_are_validated() {
        for src in [
            "[b, a] = butter(0, 0.5);",
            "[b, a] = butter(2, 1.5);",
            "[b, a] = butter(2, 0);",
            "[b, a] = butter(2, [0.5 0.2]);",
            "[b, a] = butter(2, [0.2 0.4], 'high');",
            "[b, a] = butter(2, 0.3, 'nope');",
            "[b, a] = cheby1(3, 0, 0.4);",
            "[b, a] = cheby2(3, -5, 0.4);",
        ] {
            assert!(crate::exec(src).is_err(), "{src}");
        }
        assert!(crate::exec("b = butter(2, 0.3);").is_ok());
    }

    #[test]
    fn designed_filter_runs_through_filter() {
        let src = "[b, a] = butter(3, 0.2); x = ones(1, 200); y = filter(b, a, x);";
        let y = reals(src, "y");
        assert!((y[199] - 1.0).abs() < 1e-6);
        let src = "[b, a] = butter(3, 0.2); x = 3 * ones(1, 40); y = filtfilt(b, a, x);";
        assert_close(&reals(src, "y"), &[3.0; 40], 1e-9);
    }
}
