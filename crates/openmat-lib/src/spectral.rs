//! Spectral analysis: periodogram, Welch estimates, spectrogram and tone metrics.

use std::f64::consts::PI;
use std::sync::Arc;

use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Matrix, Session, Value};
use rustfft::{Fft, FftPlanner};

use crate::sigproc::bessel_i0;
use crate::util::{col, matrix_arg, need, real_scalar, row};

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
        ("stft", stft),
        ("istft", istft),
        ("pspectrum", pspectrum),
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
                "The number of samples to overlap must be less than the length of the segments.",
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
                "The length of the segments cannot be greater than the length of the input signal.",
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

pub(crate) fn outputs(session: &Session, values: Vec<Value>) -> Value {
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
        return Err(MatError::new(
            "The number of rows of the two inputs must be equal to calculate the cross spectral density.",
        ));
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
    let unit = p.sample_rate();
    let times: Vec<f64> = (0..count)
        .map(|i| (i * step) as f64 / unit + p.window.len() as f64 / 2.0 / unit)
        .collect();
    Ok(outputs(
        session,
        vec![
            Value::Matrix(s),
            col(p.frequencies(onesided)),
            row(times),
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
    let range = range_of(2)?;
    // MATLAB integrates a Hamming-windowed periodogram of length n over a band, the total power is unwindowed.
    let p = if range.is_some() {
        Params {
            window: hamming(x.len()),
            noverlap: 0,
            nfft: x.len(),
            fs,
        }
    } else {
        Params::new(x.len(), None, None, None, fs, true)?
    };
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
        range,
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
    spur: f64,
}

/// One-sided power spectrum with a Kaiser window and the powers of the fundamental and its harmonics.
fn analyze_tones(x: &[f64], harmonics: usize) -> MatResult<Tones> {
    let n = x.len();
    if n < 128 {
        return Err(MatError::new("the signal needs at least 128 samples"));
    }
    let window = kaiser_window(n, KAISER_BETA);
    let sum: f64 = window.iter().sum();
    // Removing the windowed mean empties the DC lobe, so low-frequency tones stay detectable.
    let mean = x.iter().zip(&window).map(|(v, w)| v * w).sum::<f64>() / sum;
    let mut buffer: Vec<Complex64> = x
        .iter()
        .zip(&window)
        .map(|(v, w)| Complex64::new((v - mean) * w, 0.0))
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
    let first = 1;
    let span = |k: usize| k.saturating_sub(lobe).max(first)..=(k + lobe).min(half);
    let k0 = (first..=half)
        .max_by(|a, b| power[*a].total_cmp(&power[*b]))
        .ok_or_else(|| MatError::new("the signal is too short to analyze"))?;
    let mut used = vec![false; half + 1];
    let mut fundamental = 0.0;
    let mut weighted_bin = 0.0;
    for k in span(k0) {
        fundamental += power[k];
        weighted_bin += k as f64 * power[k];
        used[k] = true;
    }
    let f0 = weighted_bin / fundamental;
    let mut harmonic_power = 0.0;
    // `harmonics` counts the fundamental, and harmonics above Nyquist are not folded back.
    for h in 2..=harmonics {
        let bin = (h as f64 * f0).round() as usize;
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
    let spur_bin = (first..=half)
        .filter(|k| !fundamental_lobe.contains(k))
        .max_by(|a, b| power[*a].total_cmp(&power[*b]));
    let spur = spur_bin.map_or(0.0, |kp| {
        span(kp)
            .filter(|k| !fundamental_lobe.contains(k))
            .map(|k| power[k])
            .sum()
    });
    Ok(Tones {
        fundamental,
        harmonics: harmonic_power,
        total,
        spur,
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
    Ok(db(t.fundamental / t.spur.max(t.fundamental * 1e-30)))
}

// Short-time Fourier transform and pspectrum

pub(crate) fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    let denominator = bessel_i0(beta);
    (0..n)
        .map(|i| {
            let r = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
            bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denominator
        })
        .collect()
}

fn hann_periodic(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos())
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Range {
    Centered,
    TwoSided,
    OneSided,
}

struct StftOptions {
    fs: Option<f64>,
    window: Option<Vec<f64>>,
    overlap: Option<usize>,
    nfft: Option<usize>,
    range: Range,
    conjugate_symmetric: bool,
}

/// Parses `(x, [fs], 'Name', value, ...)` after the first argument.
fn stft_options(args: &[Value]) -> MatResult<StftOptions> {
    let mut options = StftOptions {
        fs: None,
        window: None,
        overlap: None,
        nfft: None,
        range: Range::Centered,
        conjugate_symmetric: false,
    };
    let mut rest = &args[1..];
    if let Some(first) = rest.first() {
        if !matches!(first, Value::Char(_)) {
            let m = first.clone().into_matrix()?;
            if m.numel() > 0 {
                let fs = m.real_scalar()?;
                if fs <= 0.0 {
                    return Err(MatError::new("the sample rate must be positive"));
                }
                options.fs = Some(fs);
            }
            rest = &rest[1..];
        }
    }
    if !rest.len().is_multiple_of(2) {
        return Err(MatError::new("options must be name and value pairs"));
    }
    for pair in rest.chunks(2) {
        let Value::Char(name) = &pair[0] else {
            return Err(MatError::new("option names must be strings"));
        };
        let count = |v: &Value| -> MatResult<usize> {
            let n = v.real_scalar()?;
            if n < 0.0 || n.fract() != 0.0 {
                return Err(MatError::new(format!(
                    "{name} must be a nonnegative integer"
                )));
            }
            Ok(n as usize)
        };
        match name.to_ascii_lowercase().as_str() {
            "window" => options.window = Some(pair[1].clone().into_matrix()?.reals()?),
            "overlaplength" => options.overlap = Some(count(&pair[1])?),
            "fftlength" => options.nfft = Some(count(&pair[1])?),
            "conjugatesymmetric" => options.conjugate_symmetric = pair[1].is_true()?,
            "frequencyrange" => {
                let Value::Char(range) = &pair[1] else {
                    return Err(MatError::new("FrequencyRange must be a string"));
                };
                options.range = match range.to_ascii_lowercase().as_str() {
                    "centered" => Range::Centered,
                    "twosided" => Range::TwoSided,
                    "onesided" => Range::OneSided,
                    other => {
                        return Err(MatError::new(format!("unknown frequency range '{other}'")))
                    }
                };
            }
            "method" => match &pair[1] {
                Value::Char(m) if m.eq_ignore_ascii_case("wola") => {}
                _ => return Err(MatError::new("only the 'wola' method is supported")),
            },
            other => return Err(MatError::new(format!("unknown option '{other}'"))),
        }
    }
    Ok(options)
}

/// FFT bin of each output row and its frequency.
fn stft_axis(p: &Params, range: Range) -> (Vec<usize>, Vec<f64>) {
    let df = p.sample_rate() / p.nfft as f64;
    let nonnegative = p.nfft / 2 + 1;
    let bins: Vec<usize> = match range {
        Range::Centered => (nonnegative..p.nfft).chain(0..nonnegative).collect(),
        Range::TwoSided => (0..p.nfft).collect(),
        Range::OneSided => (0..p.points(true)).collect(),
    };
    let freqs = bins
        .iter()
        .map(|&b| {
            let k = if range == Range::Centered && b >= nonnegative {
                b as f64 - p.nfft as f64
            } else {
                b as f64
            };
            k * df
        })
        .collect();
    (bins, freqs)
}

fn stft(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let x = signal(args, 0)?;
    let o = stft_options(args)?;
    let window = o.window.unwrap_or_else(|| hann_periodic(128));
    let len = window.len();
    let nfft = o.nfft.unwrap_or(len);
    if nfft < len {
        return Err(MatError::new(format!(
            "Expected FFTLength to be a scalar with value >= {len}."
        )));
    }
    let overlap = o.overlap.unwrap_or(len * 3 / 4);
    if overlap >= len {
        return Err(MatError::new(
            "The number of samples to overlap must be less than the length of the window.",
        ));
    }
    if x.len() < len {
        return Err(MatError::new(format!(
            "Input signal length must be greater than or equal to the window length, {len}."
        )));
    }
    let p = Params {
        window,
        noverlap: overlap,
        nfft,
        fs: o.fs,
    };
    let hop = len - overlap;
    let frames = (x.len() - overlap) / hop;
    let (bins, freqs) = stft_axis(&p, o.range);
    let plan = p.plan();
    let mut s = Matrix::zeros(bins.len(), frames);
    for i in 0..frames {
        let spectrum = p.transform(&x[i * hop..i * hop + len], &plan);
        for (row, &bin) in bins.iter().enumerate() {
            s.set(row, i, spectrum[bin]);
        }
    }
    let unit = o.fs.unwrap_or(1.0);
    let times: Vec<f64> = (0..frames)
        .map(|i| ((i * hop) as f64 + len as f64 / 2.0) / unit)
        .collect();
    Ok(outputs(
        session,
        vec![Value::Matrix(s), col(freqs), col(times)],
    ))
}

fn istft(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let s = matrix_arg(args, 0)?;
    let o = stft_options(args)?;
    if s.numel() == 0 {
        return Err(MatError::new("the spectrogram must not be empty"));
    }
    let rows = s.rows;
    let mut nfft = match o.range {
        Range::OneSided => 2 * (rows - 1),
        _ => rows,
    };
    if let (Range::OneSided, Some(w)) = (o.range, &o.window) {
        if w.len() == 2 * rows - 1 {
            nfft = w.len();
        }
    }
    let window = o.window.unwrap_or_else(|| hann_periodic(nfft));
    let len = window.len();
    if len > nfft {
        return Err(MatError::new(
            "the window must not be longer than the FFT length",
        ));
    }
    let overlap = o.overlap.unwrap_or(len * 3 / 4);
    if overlap >= len {
        return Err(MatError::new(
            "The number of samples to overlap must be less than the length of the window.",
        ));
    }
    let hop = len - overlap;
    let p = Params {
        window,
        noverlap: overlap,
        nfft,
        fs: o.fs,
    };
    let (bins, _) = stft_axis(&p, o.range);
    if bins.len() != rows {
        return Err(MatError::new(
            "the number of rows does not match the FFT length and frequency range",
        ));
    }
    let inverse = FftPlanner::new().plan_fft_inverse(nfft);
    let total = (s.cols - 1) * hop + len;
    let mut numerator = vec![Complex64::new(0.0, 0.0); total];
    let mut weights = vec![0.0; total];
    for frame in 0..s.cols {
        let mut spectrum = vec![Complex64::new(0.0, 0.0); nfft];
        for (row, &bin) in bins.iter().enumerate() {
            spectrum[bin] = s.get(row, frame);
        }
        if o.range == Range::OneSided {
            for k in rows..nfft {
                spectrum[k] = spectrum[nfft - k].conj();
            }
        }
        inverse.process(&mut spectrum);
        for j in 0..len {
            numerator[frame * hop + j] += spectrum[j] / nfft as f64 * p.window[j];
            weights[frame * hop + j] += p.window[j] * p.window[j];
        }
    }
    let y: Vec<Complex64> = numerator
        .iter()
        .zip(&weights)
        .map(|(v, w)| {
            if *w > 1e-16 {
                v / *w
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();
    let scale = y.iter().map(|z| z.norm()).fold(0.0, f64::max);
    let imag = y.iter().map(|z| z.im.abs()).fold(0.0, f64::max);
    let real = o.conjugate_symmetric || o.range == Range::OneSided || imag <= 1e-10 * scale;
    let unit = o.fs.unwrap_or(1.0);
    let times: Vec<f64> = (0..total).map(|i| i as f64 / unit).collect();
    let y = if real {
        col(y.iter().map(|z| z.re).collect())
    } else {
        complex_col(y)
    };
    Ok(outputs(session, vec![y, col(times)]))
}

const SPECTRUM_POINTS: usize = 4096;

/// One-sided power spectrum on a grid of 4096 points with a Kaiser window of beta `40 (1 - leakage)`.
fn pspectrum(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let x = real_signal(args, 0)?;
    let mut fs = None;
    let mut rest = &args[1..];
    if let Some(Value::Matrix(m)) = rest.first() {
        if m.numel() > 0 {
            let value = m.real_scalar()?;
            if value <= 0.0 {
                return Err(MatError::new("the sample rate must be positive"));
            }
            fs = Some(value);
        }
        rest = &rest[1..];
    }
    let unit = fs.unwrap_or(2.0 * PI);
    let mut leakage = 0.5;
    let mut limits = (0.0, unit / 2.0);
    let mut index = 0;
    while index < rest.len() {
        match &rest[index] {
            Value::Char(name) if name.eq_ignore_ascii_case("power") => index += 1,
            Value::Char(name) if name.eq_ignore_ascii_case("leakage") => {
                let value = rest
                    .get(index + 1)
                    .ok_or_else(|| MatError::new("Leakage needs a value"))?;
                leakage = value.real_scalar()?;
                if !(0.0..=1.0).contains(&leakage) {
                    return Err(MatError::new("Leakage must be between 0 and 1"));
                }
                index += 2;
            }
            Value::Char(name) if name.eq_ignore_ascii_case("frequencylimits") => {
                let value = rest
                    .get(index + 1)
                    .ok_or_else(|| MatError::new("FrequencyLimits needs a value"))?
                    .clone()
                    .into_matrix()?
                    .reals()?;
                if value.len() != 2
                    || value[0] >= value[1]
                    || value[0] < 0.0
                    || value[1] > unit / 2.0
                {
                    return Err(MatError::new(
                        "FrequencyLimits must be [f1 f2] with 0 <= f1 < f2 <= fs/2",
                    ));
                }
                limits = (value[0], value[1]);
                index += 2;
            }
            _ => return Err(MatError::new("unsupported pspectrum option")),
        }
    }
    let n = x.len();
    let window = kaiser_window(n, 40.0 * (1.0 - leakage));
    let sum: f64 = window.iter().sum();
    let weighted: Vec<f64> = x.iter().zip(&window).map(|(v, w)| v * w).collect();
    let step = (limits.1 - limits.0) / (SPECTRUM_POINTS - 1) as f64;
    let mut freqs = Vec::with_capacity(SPECTRUM_POINTS);
    let mut power = Vec::with_capacity(SPECTRUM_POINTS);
    for k in 0..SPECTRUM_POINTS {
        let f = limits.0 + k as f64 * step;
        let omega = 2.0 * PI * f / unit;
        let (mut re, mut im) = (0.0, 0.0);
        for (i, v) in weighted.iter().enumerate() {
            let (sin, cos) = (omega * i as f64).sin_cos();
            re += v * cos;
            im -= v * sin;
        }
        freqs.push(f);
        power.push(2.0 * (re * re + im * im) / (sum * sum));
    }
    Ok(outputs(session, vec![col(power), col(freqs)]))
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

    const REFERENCE_SETUP: &str = "fs = 1000; n = 0:999; t = n/fs; x = sin(2*pi*50*t) + 0.5*sin(2*pi*120.5*t) + 0.2*cos(2*pi*300*t + 0.3); y = filter([1 0.5 0.25], 1, x); n2 = 0:1023; t2 = n2/1024; z1 = sin(2*pi*100*t2) + 0.01*sin(2*pi*300*t2); z2 = sin(2*pi*100*t2) + 0.01*sin(2*pi*237*t2); z3 = sin(2*pi*100.3*t2) + 0.01*sin(2*pi*300.9*t2) + 0.002*sin(2*pi*237*t2);";

    /// Values from MATLAB R2025a (Signal Processing Toolbox 25.1).
    #[test]
    fn matches_matlab_reference() {
        let cases: &[(&str, &[f64])] = &[
            (
                r#"[p, f] = periodogram(x); v = [numel(p) p(1:3)' p(end) max(p) f(2) f(end)];"#,
                &[
                    513.0,
                    0.000251505866218608,
                    0.000572394323971207,
                    0.000779683124866285,
                    6.29465832429237e-06,
                    70.3699204811591,
                    0.00613592315154256,
                    std::f64::consts::PI,
                ],
            ),
            (
                r#"[p, f] = periodogram(x, hamming(1000), 1024.0, fs); v = [p(50:54)' f(2) sum(p)*f(2)];"#,
                &[
                    1.09014100511833e-05,
                    0.0326364132131006,
                    0.34527327306888,
                    0.133523587715592,
                    0.00040445916755512,
                    0.9765625,
                    0.644997934222497,
                ],
            ),
            (
                r#"[p, f] = periodogram(x, [], 1000.0, fs, 'power'); v = [max(p) p(51:53)'];"#,
                &[
                    0.500004690717393,
                    0.500004690717393,
                    4.7738200321993e-06,
                    4.8608039660983e-06,
                ],
            ),
            (
                r#"p = periodogram(x, [], 1000.0, fs, 'twosided'); v = [numel(p) p(1:3)' p(end-2:end)'];"#,
                &[
                    1000.0,
                    1.58025796329431e-06,
                    1.58048635255178e-06,
                    1.58117180844772e-06,
                    1.58231519617742e-06,
                    1.58117180844772e-06,
                    1.58048635255178e-06,
                ],
            ),
            (
                r#"p = periodogram(x + 1i*sin(2*pi*80*t), [], 1000.0, fs); v = [numel(p) p(1:3)'];"#,
                &[
                    1000.0,
                    1.58025796329422e-06,
                    1.58048635255179e-06,
                    1.58117180844758e-06,
                ],
            ),
            (
                r#"[p, f] = pwelch(x); v = [numel(p) p(1:5)' p(end) f(2)];"#,
                &[
                    129.0,
                    2.56172145123871e-05,
                    9.8484408026377e-05,
                    0.000209459626028437,
                    0.000308881819852481,
                    0.000321468653930757,
                    2.86594474095543e-06,
                    0.0245436926061703,
                ],
            ),
            (
                r#"[p, f] = pwelch(x, 250.0, 125.0, 256.0, fs); v = [p(1:5)' p(end-2:end)' f(2)];"#,
                &[
                    1.38253754313724e-06,
                    2.78830799699045e-06,
                    2.85874503511567e-06,
                    2.97840598686624e-06,
                    3.14971352300683e-06,
                    5.9063314787832e-08,
                    5.95966047411213e-08,
                    2.98879502190759e-08,
                    3.90625,
                ],
            ),
            (
                r#"[p, f] = pwelch(x, hann(200), 50.0, 512.0, fs, 'power'); v = [p(1:4)' max(p)];"#,
                &[
                    6.51919542817367e-11,
                    1.87081891093625e-07,
                    8.37644534011571e-08,
                    7.24483671824886e-08,
                    0.4846389427641,
                ],
            ),
            (
                r#"[p, f] = pwelch(x, 100.0, [], 128.0, fs, 'twosided'); v = [numel(p) p(1:3)' p(end-1:end)'];"#,
                &[
                    128.0,
                    4.4776755692366e-09,
                    1.57824200021661e-06,
                    2.97080309928125e-06,
                    2.97080309928125e-06,
                    1.57824200021661e-06,
                ],
            ),
            (
                r#"[c, f] = cpsd(x, y, 250.0, 125.0, 256.0, fs); v = [abs(c(1:5))' angle(c(1:5))'];"#,
                &[
                    2.33464680179517e-06,
                    4.70925365713641e-06,
                    4.83034798567628e-06,
                    5.03589150866665e-06,
                    5.3298160500961e-06,
                    0.0,
                    0.0230263199622315,
                    0.0453983193645828,
                    0.0665155316529803,
                    0.0858712914153653,
                ],
            ),
            (
                r#"[m, f] = mscohere(x, y, 250.0, 125.0, 256.0, fs); v = [m(1:5)' m(end-2:end)'];"#,
                &[
                    0.976514776263339,
                    0.976921579046424,
                    0.978093465606069,
                    0.979894183166186,
                    0.982124213562924,
                    0.957899852592634,
                    0.959048240622544,
                    0.959428147701047,
                ],
            ),
            (
                r#"[h, f] = tfestimate(x, y, 250.0, 125.0, 256.0, fs); v = [abs(h(1:5))' angle(h(1:5))'];"#,
                &[
                    1.68866792325756,
                    1.68892879201987,
                    1.68967428936202,
                    1.69080089513425,
                    1.69215898879847,
                    0.0,
                    -0.0230263199622315,
                    -0.0453983193645828,
                    -0.0665155316529803,
                    -0.0858712914153653,
                ],
            ),
            (
                r#"[s, f, tt, P] = spectrogram(x, 128.0, 64.0, 128.0, fs); v = [size(s) size(tt) tt(1:3) abs(s(1:4,1))' P(1:4,1)' P(1:4,end)'];"#,
                &[
                    65.0,
                    14.0,
                    1.0,
                    14.0,
                    0.064,
                    0.128,
                    0.192,
                    0.473611407581493,
                    0.47528119500459,
                    0.473957299639617,
                    0.431640009157773,
                    4.44383224948239e-06,
                    8.95044453920822e-06,
                    8.90065107443419e-06,
                    7.38221567811058e-06,
                    2.63165577859264e-06,
                    5.28433360815235e-06,
                    5.15665443936893e-06,
                    3.84086411088107e-06,
                ],
            ),
            (
                r#"[s, f, tt] = spectrogram(x); v = [size(s) size(tt) tt(1:3)];"#,
                &[
                    129.0,
                    8.0,
                    1.0,
                    8.0,
                    17.6661986832004,
                    35.3323973664008,
                    52.9985960496011,
                ],
            ),
            (
                r#"v = [bandpower(x) bandpower(x, fs, [40 60.0]) bandpower(x, fs, [100 140.0])];"#,
                &[0.6448733080697, 0.500001584191628, 0.124980712705474],
            ),
            (
                r#"[p, f] = periodogram(x, [], 1000.0, fs); v = [bandpower(p, f, [40 60.0], 'psd') bandpower(p, f, 'psd')];"#,
                &[0.500099951535518, 0.6448733080697],
            ),
            (
                r#"v = [enbw(hamming(128)) enbw(hann(1024), fs) enbw(kaiser(256, 5))];"#,
                &[1.37053150700832, 1.46627565982405, 1.36360110404547],
            ),
            (
                r#"v = [snr(z2) sfdr(z2) sinad(z2)];"#,
                &[40.0000000000002, 40.0000000000001, 40.0000000000002],
            ),
            (
                r#"v = [snr(z3) thd(z3) sfdr(z3) sinad(z3) thd(z3, 1024.0, 3)];"#,
                &[
                    53.9794000867117,
                    -39.9999999999999,
                    39.9999999999999,
                    39.8296666070117,
                    -39.9999999999999,
                ],
            ),
            (
                r#"v = [thd(z1) sfdr(z1) sinad(z1)];"#,
                &[-39.9999999999993, 39.9999999999993, 39.9999999999993],
            ),
        ];
        for (code, want) in cases {
            let got = reals(&format!("{REFERENCE_SETUP} {code}"), "v");
            assert_eq!(got.len(), want.len(), "{code}");
            for (g, w) in got.iter().zip(*want) {
                assert!(
                    (g - w).abs() <= 1e-7 * w.abs() + 1e-12,
                    "{code}: {got:?} vs {want:?}"
                );
            }
        }
    }

    #[test]
    fn errors_match_matlab_messages() {
        for (src, text) in [
            ("p = pwelch(ones(1, 1000), 2000);", "The length of the segments cannot be greater than the length of the input signal."),
            ("p = pwelch(ones(1, 1000), 100, 100);", "The number of samples to overlap must be less than the length of the segments."),
            ("c = mscohere([1 2 3 4], [1 2 3]);", "The number of rows of the two inputs must be equal to calculate the cross spectral density."),
        ] {
            let err = crate::exec(src).err().unwrap().to_string();
            assert!(err.contains(text), "{src}: {err}");
        }
        assert!(crate::exec("p = periodogram([1 2 3 4], [], 0);").is_err());
    }

    const STFT_SETUP: &str = "n = (0:999)'; xs = sin(2*pi*50*n/1000); x2 = xs + 0.3*cos(2*pi*123.4*n/1000 + 0.5); x3 = x2 + 1i*cos(2*pi*10*n/1000);";

    /// Values from MATLAB R2025a (Signal Processing Toolbox 25.1).
    #[test]
    fn stft_istft_pspectrum_match_matlab_reference() {
        let cases: &[(&str, &[f64])] = &[
            (
                r#"[s, f, t] = stft(x2, 1000); v = [size(s) f(1:3)' f(end) t(1:3)' t(end) abs(s(1:4,1))' angle(s(1:4,1))' abs(s(60:66,3))'];"#,
                &[
                    128.0,
                    28.0,
                    -492.1875,
                    -484.375,
                    -476.5625,
                    500.0,
                    0.064,
                    0.096,
                    0.128,
                    0.928,
                    1.88238272224841e-05,
                    1.92612625111453e-05,
                    1.99777172032962e-05,
                    2.09571971463274e-05,
                    0.109892866800951,
                    0.216581439218997,
                    0.317361636550117,
                    0.410351138752962,
                    0.845273546282068,
                    0.26623366162958,
                    0.115563120772834,
                    0.0602600010709044,
                    0.0438693371480315,
                    0.0602600010709044,
                    0.115563120772834,
                ],
            ),
            (
                r#"[s, f, t] = stft(x2); v = [size(s) f(1:2)' f(end) t(1:2)' abs(s(64:66,2))'];"#,
                &[
                    128.0,
                    28.0,
                    -3.09250526837745,
                    -3.04341788316511,
                    std::f64::consts::PI,
                    64.0,
                    96.0,
                    0.0728812735816156,
                    0.0854563958432042,
                    0.134534477356345,
                ],
            ),
            (
                r#"[s, f, t] = stft(x2, 1000.0, 'FrequencyRange', 'onesided'); v = [size(s) f(end) abs(s(1:3,2))' angle(s(1:3,2))'];"#,
                &[
                    65.0,
                    28.0,
                    500.0,
                    0.0728812735816156,
                    0.0854563958432042,
                    0.134534477356345,
                    0.0,
                    0.141840832978782,
                    0.238360066816173,
                ],
            ),
            (
                r#"[s, f, t] = stft(x2, 1000.0, 'FrequencyRange', 'twosided'); v = [size(s) f(end) abs(s(126:128,2))'];"#,
                &[
                    128.0,
                    28.0,
                    992.1875,
                    0.280354146920777,
                    0.134534477356345,
                    0.0854563958432042,
                ],
            ),
            (
                r#"[s, f, t] = stft(x2, 1000.0, 'Window', hamming(64), 'OverlapLength', 32.0, 'FFTLength', 256); v = [size(s) f(1:2)' t(1:2)' abs(s(126:130,3))'];"#,
                &[
                    256.0,
                    30.0,
                    -496.09375,
                    -492.1875,
                    0.032,
                    0.064,
                    0.0947492504681268,
                    0.123648059483513,
                    0.139082624416103,
                    0.123648059483513,
                    0.0947492504681268,
                ],
            ),
            (
                r#"[s, f] = stft(x2(1:990), 1000.0, 'FFTLength', 131); v = [size(s) f(1) f(end) abs(s(66:68,1))'];"#,
                &[
                    131.0,
                    27.0,
                    -496.18320610687,
                    496.18320610687,
                    0.0711245773327338,
                    0.0838171500939229,
                    0.132604665894381,
                ],
            ),
            (
                r#"[s, f] = stft(x3, 1000); v = [size(s) abs(s(60:66,2))'];"#,
                &[
                    128.0,
                    28.0,
                    1.13996201195534,
                    2.39880762541502,
                    22.7868224459111,
                    29.7324649949843,
                    18.6057677753849,
                    29.6648917587719,
                    22.9121480887276,
                ],
            ),
            (
                r#"s = stft(x2, 1000); [y, ty] = istft(s, 1000); v = [size(y) y(1:4)' y(500:503)' ty(end) max(abs(y(130:870) - x2(130:870)))];"#,
                &[
                    992.0,
                    1.0,
                    0.0,
                    0.396368470455256,
                    0.44927975820218,
                    0.523829937442076,
                    -0.475801719604962,
                    0.0554318570929811,
                    0.554978316300546,
                    0.883674484012258,
                    0.991,
                    4.44089209850063e-16,
                ],
            ),
            (
                r#"s = stft(x2, 1000.0, 'FrequencyRange', 'onesided'); y = istft(s, 1000.0, 'FrequencyRange', 'onesided'); v = [numel(y) y(1:3)' max(abs(y(130:870) - x2(130:870)))];"#,
                &[
                    992.0,
                    0.0,
                    0.396368470455256,
                    0.44927975820218,
                    4.44089209850063e-16,
                ],
            ),
            (
                r#"s = stft(x3, 1000); y = istft(s, 1000); v = [numel(y) real(y(500)) imag(y(500)) max(abs(y(130:870) - x3(130:870)))];"#,
                &[
                    992.0,
                    -0.475801719604962,
                    0.998026728428272,
                    6.68442777728834e-16,
                ],
            ),
            (
                r#"[p, f] = pspectrum(x2, 1000); v = [numel(p) f(2) f(end) max(p) p(408:413)' p(1008:1013)'];"#,
                &[
                    4096.0,
                    0.122100122100122,
                    500.0,
                    0.499128840896729,
                    0.478659454995499,
                    0.492212646451621,
                    0.499128840896729,
                    0.499128840877235,
                    0.492212646396448,
                    0.47865945491393,
                    0.0410106404566848,
                    0.0428539216197012,
                    0.0441573778106755,
                    0.0448691969319342,
                    0.044960682046352,
                    0.0444281246812654,
                ],
            ),
            (
                r#"[p, f] = pspectrum(xs, 1000.0, 'Leakage', 0.2); v = [max(p) p(408:411)'];"#,
                &[
                    0.499444610832346,
                    0.486295973091349,
                    0.495023308174643,
                    0.499444610832165,
                    0.499444610832346,
                ],
            ),
            (
                r#"[p, f] = pspectrum(x2, 1000.0, 'FrequencyLimits', [0 200.0]); v = [numel(p) f(end) max(p)];"#,
                &[4096.0, 200.0, 0.499965125598576],
            ),
            (
                r#"[p, f] = pspectrum(x2); v = [f(2) f(end) max(p)];"#,
                &[
                    0.000767177693184321,
                    std::f64::consts::PI,
                    0.499128840896729,
                ],
            ),
        ];
        for (code, want) in cases {
            let got = reals(&format!("{STFT_SETUP} {code}"), "v");
            assert_eq!(got.len(), want.len(), "{code}");
            for (g, w) in got.iter().zip(*want) {
                assert!(
                    (g - w).abs() <= 1e-7 * w.abs() + 1e-9,
                    "{code}: {got:?} vs {want:?}"
                );
            }
        }
    }

    #[test]
    fn stft_errors_match_matlab_messages() {
        for (src, text) in [
            (
                "s = stft(ones(100, 1), 1000);",
                "Input signal length must be greater than or equal to the window length, 128.",
            ),
            (
                "s = stft(ones(1000, 1), 1000, 'FFTLength', 64);",
                "Expected FFTLength to be a scalar with value >= 128.",
            ),
        ] {
            let err = crate::exec(src).err().unwrap().to_string();
            assert!(err.contains(text), "{src}: {err}");
        }
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
