//! Spectral analysis: periodogram, Welch estimates, spectrogram and tone metrics.

use std::f64::consts::PI;
use std::sync::Arc;

use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Matrix, Session, Value};
use rustfft::{Fft, FftPlanner};

use crate::sigproc::bessel_i0;
use crate::util::{col, matrix_arg, need, real_scalar};

pub fn register(session: &mut Session) {
    for (name, func) in [
        (
            "periodogram",
            periodogram as fn(&mut Session, &[Value]) -> MatResult<Value>,
        ),
        ("pwelch", pwelch),
        ("cpsd", cpsd),
        ("mscohere", mscohere),
        ("tfestimate", tfestimate),
        ("spectrogram", spectrogram),
        ("bandpower", bandpower),
        ("enbw", enbw),
        ("snr", snr),
        ("thd", thd),
        ("sfdr", sfdr),
        ("sinad", sinad),
    ] {
        session.register_builtin(name, func);
    }
}

// Argument handling

#[derive(Default)]
struct Flags {
    onesided: Option<bool>,
    power: bool,
}

/// Splits the arguments after `from` into numeric slots (an empty matrix is `None`) and string flags.
fn split_args(args: &[Value], from: usize) -> MatResult<(Vec<Option<Matrix>>, Flags)> {
    let mut slots = Vec::new();
    let mut flags = Flags::default();
    for arg in args.iter().skip(from) {
        match arg {
            Value::Char(flag) => match flag.to_ascii_lowercase().as_str() {
                "onesided" => flags.onesided = Some(true),
                "twosided" => flags.onesided = Some(false),
                "power" => flags.power = true,
                "psd" => flags.power = false,
                other => return Err(MatError::new(format!("unknown option '{other}'"))),
            },
            other => {
                let m = other.clone().into_matrix()?;
                slots.push((m.numel() > 0).then_some(m));
            }
        }
    }
    Ok((slots, flags))
}

fn slot(slots: &[Option<Matrix>], index: usize) -> Option<&Matrix> {
    slots.get(index).and_then(|m| m.as_ref())
}

fn count_slot(slots: &[Option<Matrix>], index: usize, what: &str) -> MatResult<Option<usize>> {
    slot(slots, index)
        .map(|m| {
            let v = m.real_scalar()?;
            if v < 0.0 || v.fract() != 0.0 {
                return Err(MatError::new(format!(
                    "{what} must be a nonnegative integer"
                )));
            }
            Ok(v as usize)
        })
        .transpose()
}

fn fs_slot(slots: &[Option<Matrix>], index: usize) -> MatResult<Option<f64>> {
    slot(slots, index)
        .map(|m| {
            let fs = m.real_scalar()?;
            if fs <= 0.0 {
                return Err(MatError::new("the sample rate must be positive"));
            }
            Ok(fs)
        })
        .transpose()
}

fn signal(args: &[Value], index: usize) -> MatResult<Vec<Complex64>> {
    let m = matrix_arg(args, index)?;
    if m.rows != 1 && m.cols != 1 {
        return Err(MatError::new("the input must be a vector"));
    }
    if m.numel() == 0 {
        return Err(MatError::new("the input must not be empty"));
    }
    Ok(m.data)
}

fn real_signal(args: &[Value], index: usize) -> MatResult<Vec<f64>> {
    let data = signal(args, index)?;
    if data.iter().any(|z| z.im != 0.0) {
        return Err(MatError::new("the input must be real"));
    }
    Ok(data.iter().map(|z| z.re).collect())
}

fn is_real(x: &[Complex64]) -> bool {
    x.iter().all(|z| z.im == 0.0)
}

fn hamming(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}

/// Window, overlap, transform length and sample rate of one estimate.
struct Params {
    window: Vec<f64>,
    noverlap: usize,
    nfft: usize,
    fs: Option<f64>,
}

impl Params {
    /// `rect` selects the single-segment periodogram defaults, otherwise eight Hamming segments.
    fn new(
        n: usize,
        window: Option<&Matrix>,
        noverlap: Option<usize>,
        nfft: Option<usize>,
        fs: Option<f64>,
        rect: bool,
    ) -> MatResult<Self> {
        let default_len = if rect {
            n
        } else {
            ((n as f64 / 4.5).floor() as usize).max(1)
        };
        let window = match window {
            None if rect => vec![1.0; n],
            None => hamming(default_len),
            Some(m) if m.numel() == 1 && !rect => {
                let len = m.real_scalar()?;
                if len < 1.0 || len.fract() != 0.0 {
                    return Err(MatError::new(
                        "the window length must be a positive integer",
                    ));
                }
                hamming(len as usize)
            }
            Some(m) => m.reals()?,
        };
        let len = window.len();
        if len == 0 {
            return Err(MatError::new("the window must not be empty"));
        }
        if rect && len != n {
            return Err(MatError::new(
                "the window must have the same length as the signal",
            ));
        }
        let noverlap = if rect { 0 } else { noverlap.unwrap_or(len / 2) };
        if noverlap >= len {
            return Err(MatError::new(
                "the overlap must be less than the window length",
            ));
        }
        let nfft = nfft.unwrap_or_else(|| 256.max(len.next_power_of_two()));
        if nfft == 0 {
            return Err(MatError::new("nfft must be a positive integer"));
        }
        Ok(Self {
            window,
            noverlap,
            nfft,
            fs,
        })
    }

    fn sample_rate(&self) -> f64 {
        self.fs.unwrap_or(2.0 * PI)
    }

    fn points(&self, onesided: bool) -> usize {
        match (onesided, self.nfft % 2) {
            (false, _) => self.nfft,
            (true, 0) => self.nfft / 2 + 1,
            (true, _) => self.nfft.div_ceil(2),
        }
    }

    fn frequencies(&self, onesided: bool) -> Vec<f64> {
        let step = self.sample_rate() / self.nfft as f64;
        (0..self.points(onesided))
            .map(|k| k as f64 * step)
            .collect()
    }

    /// Scales raw `|X|^2` values to PSD or power units and folds one-sided spectra.
    fn scale(&self, raw: &[Complex64], power: bool, onesided: bool) -> Vec<Complex64> {
        let sum: f64 = self.window.iter().sum();
        let sum_sq: f64 = self.window.iter().map(|w| w * w).sum();
        let factor = if power {
            1.0 / (sum * sum)
        } else {
            1.0 / (self.sample_rate() * sum_sq)
        };
        (0..self.points(onesided))
            .map(|k| {
                let nyquist = self.nfft.is_multiple_of(2) && k == self.nfft / 2;
                let fold = if onesided && k > 0 && !nyquist {
                    2.0
                } else {
                    1.0
                };
                raw[k] * (factor * fold)
            })
            .collect()
    }

    fn plan(&self) -> Arc<dyn Fft<f64>> {
        FftPlanner::new().plan_fft_forward(self.nfft)
    }

    /// Windowed transform of one segment, wrapped into `nfft` bins when the window is longer.
    fn transform(&self, segment: &[Complex64], plan: &Arc<dyn Fft<f64>>) -> Vec<Complex64> {
        let mut buffer = vec![Complex64::new(0.0, 0.0); self.nfft];
        for (j, (value, w)) in segment.iter().zip(&self.window).enumerate() {
            buffer[j % self.nfft] += value * w;
        }
        plan.process(&mut buffer);
        buffer
    }

    fn segment_count(&self, n: usize) -> MatResult<usize> {
        let len = self.window.len();
        if n < len {
            return Err(MatError::new(
                "the window must not be longer than the signal",
            ));
        }
        Ok((n - self.noverlap) / (len - self.noverlap))
    }
}

// Welch estimates

struct Raw {
    sxx: Vec<Complex64>,
    syy: Vec<Complex64>,
    syx: Vec<Complex64>,
}

/// Averaged two-sided `X conj(X)`, `Y conj(Y)` and `Y conj(X)` over all segments.
fn welch_raw(x: &[Complex64], y: Option<&[Complex64]>, p: &Params) -> MatResult<Raw> {
    let count = p.segment_count(x.len())?;
    let step = p.window.len() - p.noverlap;
    let plan = p.plan();
    let zero = Complex64::new(0.0, 0.0);
    let mut raw = Raw {
        sxx: vec![zero; p.nfft],
        syy: vec![zero; p.nfft],
        syx: vec![zero; p.nfft],
    };
    for i in 0..count {
        let range = i * step..i * step + p.window.len();
        let fx = p.transform(&x[range.clone()], &plan);
        let fy = y.map(|y| p.transform(&y[range], &plan));
        for k in 0..p.nfft {
            raw.sxx[k] += fx[k] * fx[k].conj();
            if let Some(fy) = &fy {
                raw.syy[k] += fy[k] * fy[k].conj();
                raw.syx[k] += fy[k] * fx[k].conj();
            }
        }
    }
    let scale = 1.0 / count as f64;
    for values in [&mut raw.sxx, &mut raw.syy, &mut raw.syx] {
        for v in values.iter_mut() {
            *v *= scale;
        }
    }
    Ok(raw)
}

fn real_col(values: &[Complex64]) -> Value {
    col(values.iter().map(|z| z.re).collect())
}

fn complex_col(values: Vec<Complex64>) -> Value {
    Value::Matrix(Matrix {
        rows: values.len(),
        cols: 1,
        data: values,
        logical: false,
    })
}

fn outputs(session: &Session, values: Vec<Value>) -> Value {
    let wanted = session.nargout.clamp(1, values.len());
    if wanted == 1 {
        values
            .into_iter()
            .next()
            .unwrap_or_else(|| Value::scalar(0.0))
    } else {
        Value::Multi(values.into_iter().take(wanted).collect())
    }
}

fn periodogram(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let x = signal(args, 0)?;
    let (slots, flags) = split_args(args, 1)?;
    let p = Params::new(
        x.len(),
        slot(&slots, 0),
        None,
        count_slot(&slots, 1, "nfft")?,
        fs_slot(&slots, 2)?,
        true,
    )?;
    let onesided = flags.onesided.unwrap_or_else(|| is_real(&x));
    let raw = welch_raw(&x, None, &p)?;
    let pxx = p.scale(&raw.sxx, flags.power, onesided);
    Ok(outputs(
        session,
        vec![real_col(&pxx), col(p.frequencies(onesided))],
    ))
}

fn pwelch(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let x = signal(args, 0)?;
    let (slots, flags) = split_args(args, 1)?;
    let p = Params::new(
        x.len(),
        slot(&slots, 0),
        count_slot(&slots, 1, "the overlap")?,
        count_slot(&slots, 2, "nfft")?,
        fs_slot(&slots, 3)?,
        false,
    )?;
    let onesided = flags.onesided.unwrap_or_else(|| is_real(&x));
    let raw = welch_raw(&x, None, &p)?;
    let pxx = p.scale(&raw.sxx, flags.power, onesided);
    Ok(outputs(
        session,
        vec![real_col(&pxx), col(p.frequencies(onesided))],
    ))
}

/// Shared setup of the two-signal estimates `(x, y, window, noverlap, nfft, fs)`.
type Pair = (Vec<Complex64>, Vec<Complex64>, Params, bool, bool);

fn pair(args: &[Value]) -> MatResult<Pair> {
    need(args, 2)?;
    let x = signal(args, 0)?;
    let y = signal(args, 1)?;
    if x.len() != y.len() {
        return Err(MatError::new("x and y must have the same length"));
    }
    let (slots, flags) = split_args(args, 2)?;
    let p = Params::new(
        x.len(),
        slot(&slots, 0),
        count_slot(&slots, 1, "the overlap")?,
        count_slot(&slots, 2, "nfft")?,
        fs_slot(&slots, 3)?,
        false,
    )?;
    let onesided = flags.onesided.unwrap_or_else(|| is_real(&x) && is_real(&y));
    Ok((x, y, p, onesided, flags.power))
}

fn cpsd(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (x, y, p, onesided, power) = pair(args)?;
    let raw = welch_raw(&x, Some(&y), &p)?;
    let pxy: Vec<Complex64> = raw.syx.iter().map(|z| z.conj()).collect();
    let pxy = p.scale(&pxy, power, onesided);
    Ok(outputs(
        session,
        vec![complex_col(pxy), col(p.frequencies(onesided))],
    ))
}

fn mscohere(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (x, y, p, onesided, _) = pair(args)?;
    let raw = welch_raw(&x, Some(&y), &p)?;
    let cxy: Vec<f64> = (0..p.points(onesided))
        .map(|k| raw.syx[k].norm_sqr() / (raw.sxx[k].re * raw.syy[k].re))
        .collect();
    Ok(outputs(
        session,
        vec![col(cxy), col(p.frequencies(onesided))],
    ))
}

fn tfestimate(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (x, y, p, onesided, _) = pair(args)?;
    let raw = welch_raw(&x, Some(&y), &p)?;
    let txy: Vec<Complex64> = (0..p.points(onesided))
        .map(|k| raw.syx[k] / raw.sxx[k].re)
        .collect();
    Ok(outputs(
        session,
        vec![complex_col(txy), col(p.frequencies(onesided))],
    ))
}

// Spectrogram

fn spectrogram(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let x = signal(args, 0)?;
    let (slots, flags) = split_args(args, 1)?;
    let p = Params::new(
        x.len(),
        slot(&slots, 0),
        count_slot(&slots, 1, "the overlap")?,
        count_slot(&slots, 2, "nfft")?,
        fs_slot(&slots, 3)?,
        false,
    )?;
    let onesided = flags.onesided.unwrap_or_else(|| is_real(&x));
    let count = p.segment_count(x.len())?;
    let step = p.window.len() - p.noverlap;
    let plan = p.plan();
    let rows = p.points(onesided);
    let mut s = Matrix::zeros(rows, count);
    let mut power = Matrix::zeros(rows, count);
    for i in 0..count {
        let spectrum = p.transform(&x[i * step..i * step + p.window.len()], &plan);
        let squared: Vec<Complex64> = spectrum
            .iter()
            .map(|z| Complex64::new(z.norm_sqr(), 0.0))
            .collect();
        let scaled = p.scale(&squared, flags.power, onesided);
        for k in 0..rows {
            s.set(k, i, spectrum[k]);
            power.set(k, i, scaled[k]);
        }
    }
    let unit = p.fs.unwrap_or(1.0);
    let times: Vec<f64> = (0..count)
        .map(|i| (i * step) as f64 / unit + p.window.len() as f64 / 2.0 / unit)
        .collect();
    Ok(outputs(
        session,
        vec![
            Value::Matrix(s),
            col(p.frequencies(onesided)),
            crate::util::row(times),
            Value::Matrix(power),
        ],
    ))
}

// Band power and window metrics

fn bandpower(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let psd_mode = args
        .iter()
        .skip(1)
        .any(|a| matches!(a, Value::Char(flag) if flag.eq_ignore_ascii_case("psd")));
    let range_of = |index: usize| -> MatResult<Option<(f64, f64)>> {
        match args.get(index) {
            Some(Value::Char(_)) | None => Ok(None),
            Some(value) => {
                let r = value.clone().into_matrix()?.reals()?;
                match r.len() {
                    0 => Ok(None),
                    2 if r[0] <= r[1] => Ok(Some((r[0], r[1]))),
                    _ => Err(MatError::new(
                        "the frequency range must be [f1 f2] with f1 <= f2",
                    )),
                }
            }
        }
    };
    let integrate = |pxx: &[f64], f: &[f64], range: Option<(f64, f64)>| -> MatResult<f64> {
        if pxx.len() != f.len() || f.len() < 2 {
            return Err(MatError::new(
                "the PSD and the frequency vector must have the same length",
            ));
        }
        let df = f[1] - f[0];
        let (lo, hi) = range.unwrap_or((f[0], f[f.len() - 1]));
        let eps = df * 1e-9;
        Ok(pxx
            .iter()
            .zip(f)
            .filter(|(_, fk)| **fk >= lo - eps && **fk <= hi + eps)
            .map(|(p, _)| p)
            .sum::<f64>()
            * df)
    };
    if psd_mode {
        need(args, 2)?;
        let pxx = matrix_arg(args, 0)?.reals()?;
        let f = matrix_arg(args, 1)?.reals()?;
        return Ok(Value::scalar(integrate(&pxx, &f, range_of(2)?)?));
    }
    let x = signal(args, 0)?;
    let fs = match args.get(1) {
        Some(Value::Matrix(m)) if m.numel() > 0 => Some(m.real_scalar()?),
        _ => None,
    };
    let p = Params::new(x.len(), None, None, None, fs, true)?;
    let onesided = is_real(&x);
    let raw = welch_raw(&x, None, &p)?;
    let pxx: Vec<f64> = p
        .scale(&raw.sxx, false, onesided)
        .iter()
        .map(|z| z.re)
        .collect();
    Ok(Value::scalar(integrate(
        &pxx,
        &p.frequencies(onesided),
        range_of(2)?,
    )?))
}

fn enbw(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let w = matrix_arg(args, 0)?.reals()?;
    let sum: f64 = w.iter().sum();
    let sum_sq: f64 = w.iter().map(|v| v * v).sum();
    if w.is_empty() || sum == 0.0 {
        return Err(MatError::new(
            "the window must be nonempty with a nonzero sum",
        ));
    }
    let scale = if args.len() >= 2 {
        real_scalar(args, 1)?
    } else {
        w.len() as f64
    };
    Ok(Value::scalar(scale * sum_sq / (sum * sum)))
}

// Tone metrics

const KAISER_BETA: f64 = 38.0;

struct Tones {
    fundamental: f64,
    harmonics: f64,
    total: f64,
    fundamental_peak: f64,
    spur_peak: f64,
}

/// One-sided power spectrum with a Kaiser window and the powers of the fundamental and its harmonics.
fn analyze_tones(x: &[f64], harmonics: usize) -> MatResult<Tones> {
    let n = x.len();
    if n < 128 {
        return Err(MatError::new("the signal needs at least 128 samples"));
    }
    let denominator = bessel_i0(KAISER_BETA);
    let window: Vec<f64> = (0..n)
        .map(|i| {
            let r = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
            bessel_i0(KAISER_BETA * (1.0 - r * r).max(0.0).sqrt()) / denominator
        })
        .collect();
    let sum: f64 = window.iter().sum();
    let mut buffer: Vec<Complex64> = x
        .iter()
        .zip(&window)
        .map(|(v, w)| Complex64::new(v * w, 0.0))
        .collect();
    FftPlanner::new().plan_fft_forward(n).process(&mut buffer);
    let half = n / 2;
    let power: Vec<f64> = (0..=half)
        .map(|k| {
            let fold = if k == 0 || (n.is_multiple_of(2) && k == half) {
                1.0
            } else {
                2.0
            };
            fold * buffer[k].norm_sqr() / (sum * sum) / 2.0
        })
        .collect();
    // Half width in bins of the Kaiser main lobe.
    let lobe = (1.0 + (KAISER_BETA / PI).powi(2)).sqrt().ceil() as usize;
    let first = lobe + 1;
    let span = |k: usize| k.saturating_sub(lobe).max(first)..=(k + lobe).min(half);
    let k0 = (first..=half)
        .max_by(|a, b| power[*a].total_cmp(&power[*b]))
        .ok_or_else(|| MatError::new("the signal is too short to analyze"))?;
    let mut used = vec![false; half + 1];
    let mut fundamental = 0.0;
    for k in span(k0) {
        fundamental += power[k];
        used[k] = true;
    }
    let mut harmonic_power = 0.0;
    for h in 2..=harmonics {
        let folded = (h * k0) % n;
        let bin = if folded > half { n - folded } else { folded };
        if bin < first || bin > half {
            continue;
        }
        let low = bin.saturating_sub(2).max(first);
        let high = (bin + 2).min(half);
        let kh = (low..=high)
            .max_by(|a, b| power[*a].total_cmp(&power[*b]))
            .unwrap_or(bin);
        for k in span(kh) {
            if !used[k] {
                harmonic_power += power[k];
                used[k] = true;
            }
        }
    }
    let total: f64 = power[first..].iter().sum();
    let fundamental_lobe = span(k0);
    let spur_peak = (first..=half)
        .filter(|k| !fundamental_lobe.contains(k))
        .map(|k| power[k])
        .fold(0.0, f64::max);
    Ok(Tones {
        fundamental,
        harmonics: harmonic_power,
        total,
        fundamental_peak: power[k0],
        spur_peak,
    })
}

fn db(ratio: f64) -> Value {
    Value::scalar(10.0 * ratio.log10())
}

fn snr(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let t = analyze_tones(&real_signal(args, 0)?, 6)?;
    Ok(db(t.fundamental
        / (t.total - t.fundamental - t.harmonics)
            .max(t.fundamental * 1e-30)))
}

fn sinad(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let t = analyze_tones(&real_signal(args, 0)?, 6)?;
    Ok(db(
        t.fundamental / (t.total - t.fundamental).max(t.fundamental * 1e-30)
    ))
}

fn thd(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let count = match args.get(2) {
        Some(_) => {
            let n = real_scalar(args, 2)?;
            if n < 1.0 || n.fract() != 0.0 {
                return Err(MatError::new(
                    "the number of harmonics must be a positive integer",
                ));
            }
            n as usize
        }
        None => 6,
    };
    let t = analyze_tones(&real_signal(args, 0)?, count)?;
    Ok(db(t.harmonics / t.fundamental))
}

fn sfdr(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let t = analyze_tones(&real_signal(args, 0)?, 6)?;
    Ok(db(t.fundamental_peak / t.spur_peak))
}

#[cfg(test)]
mod tests {
    use crate::exec;

    fn reals(src: &str, name: &str) -> Vec<f64> {
        let session = exec(src).unwrap_or_else(|err| panic!("{src}: {err}"));
        session
            .get(name)
            .unwrap_or_else(|| panic!("{name} missing"))
            .clone()
            .into_matrix()
            .unwrap()
            .reals()
            .unwrap()
    }

    fn scalar(src: &str, name: &str) -> f64 {
        reals(src, name)[0]
    }

    const SINE: &str = "n = 0:999; x = sin(2*pi*50*n/1000);";

    #[test]
    fn periodogram_of_a_sine_peaks_at_half_the_power() {
        let src = format!("{SINE} [p, f] = periodogram(x, [], 1000, 1000, 'power'); a = max(p); b = f(p == max(p));");
        assert!((scalar(&src, "a") - 0.5).abs() < 1e-9);
        assert!((scalar(&src, "b") - 50.0).abs() < 1e-9);
        let src = format!("{SINE} [p, f] = periodogram(x, [], 1000, 1000); a = sum(p) * (f(2) - f(1)); n = numel(p);");
        assert!((scalar(&src, "a") - 0.5).abs() < 1e-9);
        assert_eq!(scalar(&src, "n"), 501.0);
    }

    #[test]
    fn periodogram_defaults_and_sides() {
        let src = format!("{SINE} [p, f] = periodogram(x); a = numel(p); b = f(end);");
        assert_eq!(scalar(&src, "a"), 1024.0 / 2.0 + 1.0);
        assert!((scalar(&src, "b") - std::f64::consts::PI).abs() < 1e-12);
        let src = format!("{SINE} p = periodogram(x, [], 1000, 1000, 'twosided'); a = numel(p);");
        assert_eq!(scalar(&src, "a"), 1000.0);
    }

    #[test]
    fn pwelch_finds_the_tone_and_conserves_power() {
        let src = format!("{SINE} [p, f] = pwelch(x, 250, 125, 256, 1000); a = f(p == max(p)); b = sum(p) * (f(2) - f(1));");
        assert!((scalar(&src, "a") - 50.0).abs() < 4.0);
        assert!(
            (scalar(&src, "b") - 0.5).abs() < 0.03,
            "{}",
            scalar(&src, "b")
        );
        let src = format!("{SINE} [p, f] = pwelch(x, 250, 125, 256, 1000, 'power'); a = max(p);");
        assert!((scalar(&src, "a") - 0.5).abs() < 0.05);
    }

    #[test]
    fn pwelch_default_segmentation() {
        let src = format!("{SINE} [p, f] = pwelch(x); a = numel(p);");
        assert_eq!(scalar(&src, "a"), 129.0);
        assert!(crate::exec(&format!("{SINE} p = pwelch(x, 2000);")).is_err());
        assert!(crate::exec(&format!("{SINE} p = pwelch(x, 100, 100);")).is_err());
    }

    #[test]
    fn cross_spectra_of_scaled_copies() {
        let setup = format!("{SINE} y = 2 * x;");
        let src = format!("{setup} p = pwelch(x, 250, 125, 256, 1000); q = cpsd(x, x, 250, 125, 256, 1000); a = max(abs(p - q));");
        assert!(scalar(&src, "a") < 1e-12);
        let src = format!("{setup} c = mscohere(x, y, 250, 125, 256, 1000); a = min(c);");
        assert!((scalar(&src, "a") - 1.0).abs() < 1e-9);
        let src = format!(
            "{setup} [t, f] = tfestimate(x, y, 250, 125, 256, 1000); a = abs(t(11)); b = f(11);"
        );
        assert!(
            (scalar(&src, "a") - 2.0).abs() < 1e-9,
            "{}",
            scalar(&src, "a")
        );
        assert!(crate::exec("c = mscohere([1 2 3 4], [1 2 3]);").is_err());
    }

    #[test]
    fn spectrogram_layout() {
        let src = format!("{SINE} [s, f, t, p] = spectrogram(x, 128, 64, 128, 1000); r = size(s, 1); c = size(s, 2); a = t(1); b = f(end); q = p(:, 3); d = f(q == max(q));");
        assert_eq!(scalar(&src, "r"), 65.0);
        assert_eq!(scalar(&src, "c"), 14.0);
        assert!((scalar(&src, "a") - 0.064).abs() < 1e-12);
        assert!((scalar(&src, "b") - 500.0).abs() < 1e-12);
        assert!((scalar(&src, "d") - 54.6875).abs() < 8.0);
    }

    #[test]
    fn bandpower_and_enbw() {
        let src = format!("{SINE} a = bandpower(x, 1000, [40 60]); b = bandpower(x, 1000, [200 300]); c = bandpower(x);");
        assert!(
            (scalar(&src, "a") - 0.5).abs() < 0.01,
            "{}",
            scalar(&src, "a")
        );
        assert!(scalar(&src, "b") < 1e-3);
        assert!((scalar(&src, "c") - 0.5).abs() < 0.01);
        let src = format!(
            "{SINE} [p, f] = periodogram(x, [], 1000, 1000); a = bandpower(p, f, [40 60], 'psd');"
        );
        assert!((scalar(&src, "a") - 0.5).abs() < 1e-9);
        assert!((scalar("a = enbw(rectwin(64));", "a") - 1.0).abs() < 1e-12);
        assert!((scalar("a = enbw(hann(1024, 'periodic'));", "a") - 1.5).abs() < 1e-12);
        assert!(
            (scalar("a = enbw(hamming(128), 8);", "a")
                - 8.0 * 128.0 * 0.0
                - scalar("w = hamming(128); a = 8 * sum(w.^2) / sum(w)^2;", "a"))
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn tone_metrics_match_synthetic_levels() {
        let base = "n = 0:1023; x = sin(2*pi*100*n/1024);";
        let harmonic = format!("{base} x = x + 0.01 * sin(2*pi*300*n/1024);");
        assert!((scalar(&format!("{harmonic} a = thd(x);"), "a") + 40.0).abs() < 0.2);
        assert!((scalar(&format!("{harmonic} a = sinad(x);"), "a") - 40.0).abs() < 0.3);
        assert!((scalar(&format!("{harmonic} a = sfdr(x);"), "a") - 40.0).abs() < 0.3);
        let spur = format!("{base} x = x + 0.01 * sin(2*pi*237*n/1024);");
        assert!((scalar(&format!("{spur} a = snr(x);"), "a") - 40.0).abs() < 0.3);
        assert!((scalar(&format!("{spur} a = sfdr(x);"), "a") - 40.0).abs() < 0.3);
        assert!(scalar(&format!("{harmonic} a = snr(x);"), "a") > 80.0);
        assert!(crate::exec("a = sinad(ones(1, 10));").is_err());
    }
}
