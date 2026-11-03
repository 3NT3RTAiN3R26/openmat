//! Resampling, peak finding, median filtering and envelopes.

use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Matrix, Session, Value};
use rustfft::FftPlanner;

use crate::sigproc::{along, default_dim};
use crate::util::{col, matrix_arg, need, positive, real_scalar, row};

type Builtin = fn(&mut Session, &[Value]) -> MatResult<Value>;

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("upsample", upsample as Builtin),
        ("downsample", downsample),
        ("upfirdn", upfirdn),
        ("interp", interp),
        ("decimate", decimate),
        ("resample", resample),
        ("findpeaks", findpeaks),
        ("medfilt1", medfilt1),
        ("envelope", envelope),
    ] {
        session.register_builtin(name, func);
    }
}

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

fn count(v: f64, what: &str) -> MatResult<usize> {
    if v < 0.0 || v.fract() != 0.0 {
        return Err(MatError::new(format!(
            "{what} must be a nonnegative integer"
        )));
    }
    Ok(v as usize)
}

fn length_along(m: &Matrix, dim: usize) -> usize {
    if dim == 1 {
        m.rows
    } else {
        m.cols
    }
}

fn reals_of(value: &Value) -> MatResult<Vec<f64>> {
    value.clone().into_matrix()?.reals()
}

/// Calls a builtin that returns `[a, b]` when asked for two outputs.
fn call_two(session: &mut Session, func: Builtin, args: &[Value]) -> MatResult<(Value, Value)> {
    let saved = session.nargout;
    session.nargout = 2;
    let result = func(session, args);
    session.nargout = saved;
    match result? {
        Value::Multi(mut v) if v.len() >= 2 => {
            let b = v.swap_remove(1);
            Ok((v.swap_remove(0), b))
        }
        _ => Err(MatError::new("internal error: expected two outputs")),
    }
}

// upsample, downsample, upfirdn

fn rate_args(args: &[Value]) -> MatResult<(Matrix, usize, usize)> {
    need(args, 2)?;
    let m = matrix_arg(args, 0)?;
    let n = positive(real_scalar(args, 1)?)?;
    let phase = if args.len() >= 3 {
        count(real_scalar(args, 2)?, "the phase offset")?
    } else {
        0
    };
    if phase >= n {
        return Err(MatError::new(
            "The phase offset must be less than the sampling factor.",
        ));
    }
    Ok((m, n, phase))
}

fn upsample(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (m, n, phase) = rate_args(args)?;
    let dim = default_dim(&m);
    let len = length_along(&m, dim) * n;
    Ok(Value::Matrix(along(&m, dim, len, |x| {
        let mut out = vec![ZERO; x.len() * n];
        for (i, v) in x.iter().enumerate() {
            out[i * n + phase] = *v;
        }
        out
    })?))
}

fn downsample(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (m, n, phase) = rate_args(args)?;
    let dim = default_dim(&m);
    let len = length_along(&m, dim).saturating_sub(phase).div_ceil(n);
    Ok(Value::Matrix(along(&m, dim, len, |x| {
        x.iter().skip(phase).step_by(n).copied().collect()
    })?))
}

/// Upsample by `p`, filter with `h` and downsample by `q`.
fn upfirdn_vec(x: &[Complex64], h: &[Complex64], p: usize, q: usize) -> Vec<Complex64> {
    if x.is_empty() || h.is_empty() {
        return Vec::new();
    }
    let length = ((x.len() - 1) * p + h.len()).div_ceil(q);
    let mut y = vec![ZERO; length];
    for (i, xv) in x.iter().enumerate() {
        for (k, hv) in h.iter().enumerate() {
            let position = i * p + k;
            if position.is_multiple_of(q) {
                y[position / q] += xv * hv;
            }
        }
    }
    y
}

fn upfirdn(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let m = matrix_arg(args, 0)?;
    let h = matrix_arg(args, 1)?;
    if h.numel() == 0 || (h.rows != 1 && h.cols != 1) {
        return Err(MatError::new("the filter must be a nonempty vector"));
    }
    let p = if args.len() >= 3 {
        positive(real_scalar(args, 2)?)?
    } else {
        1
    };
    let q = if args.len() >= 4 {
        positive(real_scalar(args, 3)?)?
    } else {
        1
    };
    let dim = default_dim(&m);
    let len = (length_along(&m, dim).saturating_sub(1) * p + h.numel()).div_ceil(q);
    Ok(Value::Matrix(along(&m, dim, len, |x| {
        upfirdn_vec(x, &h.data, p, q)
    })?))
}

// interp

fn interp(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let m = matrix_arg(args, 0)?;
    let r = positive(real_scalar(args, 1)?)?;
    let l = if args.len() >= 3 {
        positive(real_scalar(args, 2)?)?
    } else {
        4
    };
    let cutoff = if args.len() >= 4 {
        real_scalar(args, 3)?
    } else {
        0.5
    };
    if !(cutoff > 0.0 && cutoff <= 1.0) {
        return Err(MatError::new("the cutoff must be in (0, 1]"));
    }
    let dim = default_dim(&m);
    let n = length_along(&m, dim);
    if n < 2 * l + 1 {
        return Err(MatError::new(format!(
            "Length of data sequence must be at least {}.\n You either need more data or a shorter filter (L).",
            2 * l + 1
        )));
    }
    // Pass band of the original rate and the images of that band at multiples of 2/r.
    let mut bands = vec![0.0, cutoff / r as f64];
    let mut gains = vec![1.0, 1.0];
    let mut k = 1.0;
    while (2.0 * k - cutoff) / (r as f64) < 1.0 {
        bands.push((2.0 * k - cutoff) / r as f64);
        bands.push(((2.0 * k + cutoff) / r as f64).min(1.0));
        gains.extend([0.0, 0.0]);
        k += 1.0;
    }
    let order = 2 * l * r;
    let designed = crate::fir::firls(
        session,
        &[Value::scalar(order as f64), row(bands), row(gains)],
    )?;
    let b: Vec<f64> = reals_of(&designed)?.iter().map(|v| v * r as f64).collect();
    let taps: Vec<Complex64> = b.iter().map(|v| Complex64::new(*v, 0.0)).collect();
    let y = along(&m, dim, n * r, |x| {
        // Odd extension of `l` samples at both ends keeps the edge error small.
        let mut padded: Vec<Complex64> = (1..=l).rev().map(|i| 2.0 * x[0] - x[i]).collect();
        padded.extend_from_slice(x);
        padded.extend((1..=l).map(|i| 2.0 * x[x.len() - 1] - x[x.len() - 1 - i]));
        let start = 2 * l * r;
        upfirdn_vec(&padded, &taps, r, 1)[start..start + x.len() * r].to_vec()
    })?;
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![Value::Matrix(y), col(b)]))
    } else {
        Ok(Value::Matrix(y))
    }
}

// decimate

fn decimate(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let m = matrix_arg(args, 0)?;
    let r = positive(real_scalar(args, 1)?)?;
    let mut order = None;
    let mut fir = false;
    for arg in &args[2..] {
        match arg {
            Value::Char(s) if s.eq_ignore_ascii_case("fir") => fir = true,
            Value::Char(s) if s.eq_ignore_ascii_case("iir") => fir = false,
            Value::Char(other) => {
                return Err(MatError::new(format!("unknown decimate option '{other}'")))
            }
            value => order = Some(positive(value.real_scalar()?)?),
        }
    }
    if r == 1 {
        return Ok(Value::Matrix(m));
    }
    let dim = default_dim(&m);
    let n = length_along(&m, dim);
    let out_len = n.div_ceil(r);
    if fir {
        let taps_order = order.unwrap_or(30);
        let designed = crate::fir::fir1(
            session,
            &[
                Value::scalar(taps_order as f64),
                Value::scalar(1.0 / r as f64),
            ],
        )?;
        let b = reals_of(&designed)?;
        let half = taps_order / 2;
        if n <= half {
            return Err(MatError::new(
                "the signal must be longer than half the filter order",
            ));
        }
        let y = along(&m, dim, out_len, |x| {
            // Odd reflection about the end points keeps the transient small.
            let mut padded: Vec<Complex64> = (1..=half).rev().map(|i| 2.0 * x[0] - x[i]).collect();
            padded.extend_from_slice(x);
            padded.extend((1..=half).map(|i| 2.0 * x[x.len() - 1] - x[x.len() - 1 - i]));
            let valid = padded.len().saturating_sub(b.len() - 1);
            (0..valid)
                .step_by(r)
                .map(|start| {
                    b.iter()
                        .enumerate()
                        .map(|(k, bv)| padded[start + b.len() - 1 - k] * bv)
                        .sum()
                })
                .collect()
        })?;
        return Ok(Value::Matrix(y));
    }
    let n_order = order.unwrap_or(8);
    let (b, a) = call_two(
        session,
        crate::iir::cheby1,
        &[
            Value::scalar(n_order as f64),
            Value::scalar(0.05),
            Value::scalar(0.8 / r as f64),
        ],
    )?;
    let filtered = crate::filt::filtfilt(session, &[b, a, Value::Matrix(m)])?.into_matrix()?;
    // The kept samples are aligned so that the last sample survives.
    let start = (n - 1) % r;
    Ok(Value::Matrix(along(&filtered, dim, out_len, |x| {
        x.iter().skip(start).step_by(r).copied().collect()
    })?))
}

// resample

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn resample(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let m = matrix_arg(args, 0)?;
    let mut p = positive(real_scalar(args, 1)?)?;
    let mut q = positive(real_scalar(args, 2)?)?;
    let n = if args.len() >= 4 {
        positive(real_scalar(args, 3)?)?
    } else {
        10
    };
    let beta = if args.len() >= 5 {
        real_scalar(args, 4)?
    } else {
        5.0
    };
    let g = gcd(p, q);
    p /= g;
    q /= g;
    let pq = p.max(q);
    let length = 2 * n * pq + 1;
    let window = crate::spectral::kaiser_window(length, beta);
    let designed = crate::fir::fir1(
        session,
        &[
            Value::scalar((length - 1) as f64),
            Value::scalar(1.0 / pq as f64),
            col(window),
        ],
    )?;
    let mut h = reals_of(&designed)?;
    let sum: f64 = h.iter().sum();
    for v in &mut h {
        *v *= p as f64 / sum;
    }
    if p == 1 && q == 1 {
        return Ok(Value::Matrix(m));
    }
    // Pad the filter in front so the delay is a whole number of output samples.
    let half = n * pq;
    let pad = (q - half % q) % q;
    let delay = (half + pad) / q;
    let taps: Vec<Complex64> = std::iter::repeat_n(0.0, pad)
        .chain(h.iter().copied())
        .map(|v| Complex64::new(v, 0.0))
        .collect();
    let dim = default_dim(&m);
    let len = length_along(&m, dim);
    let out_len = (len * p).div_ceil(q);
    let y = along(&m, dim, out_len, |x| {
        let mut y = upfirdn_vec(x, &taps, p, q);
        y.resize(delay + out_len, ZERO);
        y[delay..delay + out_len].to_vec()
    })?;
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![Value::Matrix(y), row(h)]))
    } else {
        Ok(Value::Matrix(y))
    }
}

// findpeaks

struct Peak {
    index: usize,
    prominence: f64,
    left: f64,
    right: f64,
}

/// Indices of local maxima. A flat top counts once, at its first sample.
fn local_maxima(y: &[f64]) -> Vec<usize> {
    let mut peaks = Vec::new();
    let mut i = 1;
    while i + 1 < y.len() {
        if y[i] > y[i - 1] {
            let mut j = i;
            while j + 1 < y.len() && y[j + 1] == y[i] {
                j += 1;
            }
            if j + 1 < y.len() && y[j + 1] < y[i] {
                peaks.push(i);
            }
            i = j;
        }
        i += 1;
    }
    peaks
}

/// Prominence and the half-prominence crossings (fractional indices) of the peak at `i`.
fn measure_peak(y: &[f64], i: usize) -> Peak {
    let height = y[i];
    let mut left_min = height;
    let mut j = i;
    while j > 0 && y[j - 1] <= height {
        j -= 1;
        left_min = left_min.min(y[j]);
    }
    let mut right_min = height;
    let mut j = i;
    while j + 1 < y.len() && y[j + 1] <= height {
        j += 1;
        right_min = right_min.min(y[j]);
    }
    let prominence = height - left_min.max(right_min);
    let reference = height - prominence / 2.0;
    let left;
    let mut j = i;
    loop {
        if j == 0 {
            left = 0.0;
            break;
        }
        if y[j - 1] < reference {
            left = (j - 1) as f64 + (reference - y[j - 1]) / (y[j] - y[j - 1]);
            break;
        }
        j -= 1;
    }
    let mut right = (y.len() - 1) as f64;
    let mut j = i;
    while j + 1 < y.len() {
        if y[j + 1] < reference {
            right = j as f64 + (y[j] - reference) / (y[j] - y[j + 1]);
            break;
        }
        j += 1;
    }
    Peak {
        index: i,
        prominence,
        left,
        right,
    }
}

fn findpeaks(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let m = matrix_arg(args, 0)?;
    if m.rows != 1 && m.cols != 1 {
        return Err(MatError::new("the data must be a vector"));
    }
    let y = m.reals()?;
    let n = y.len();
    let mut rest = &args[1..];
    // Optional sample rate or location vector.
    let mut positions: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    if let Some(first) = rest.first() {
        if !matches!(first, Value::Char(_)) {
            let v = first.clone().into_matrix()?;
            if v.numel() == 1 {
                let fs = v.real_scalar()?;
                if fs <= 0.0 {
                    return Err(MatError::new("the sample rate must be positive"));
                }
                positions = (0..n).map(|i| i as f64 / fs).collect();
            } else if v.numel() == n {
                positions = v.reals()?;
            } else {
                return Err(MatError::new(
                    "X must be a scalar sample rate or have one element per sample",
                ));
            }
            rest = &rest[1..];
        }
    }
    if !rest.len().is_multiple_of(2) {
        return Err(MatError::new("options must be name and value pairs"));
    }
    let mut min_height = f64::NEG_INFINITY;
    let mut min_distance = 0.0;
    let mut threshold = 0.0;
    let mut max_peaks = usize::MAX;
    let mut sort = "none".to_string();
    let mut min_prominence = f64::NEG_INFINITY;
    let mut min_width = f64::NEG_INFINITY;
    let mut max_width = f64::INFINITY;
    for pair in rest.chunks(2) {
        let Value::Char(name) = &pair[0] else {
            return Err(MatError::new("option names must be strings"));
        };
        let number = || pair[1].real_scalar();
        match name.to_ascii_lowercase().as_str() {
            "minpeakheight" => min_height = number()?,
            "minpeakdistance" => min_distance = number()?,
            "threshold" => threshold = number()?,
            "npeaks" => max_peaks = positive(number()?)?,
            "minpeakprominence" => min_prominence = number()?,
            "minpeakwidth" => min_width = number()?,
            "maxpeakwidth" => max_width = number()?,
            "sortstr" => match &pair[1] {
                Value::Char(s) => sort = s.to_ascii_lowercase(),
                _ => return Err(MatError::new("SortStr must be a string")),
            },
            other => return Err(MatError::new(format!("unknown option '{other}'"))),
        }
    }
    if !["none", "ascend", "descend"].contains(&sort.as_str()) {
        return Err(MatError::new(
            "SortStr must be 'none', 'ascend' or 'descend'",
        ));
    }
    // Fractional index to position, interpolating between samples.
    let place = |t: f64| -> f64 {
        let low = (t.floor() as usize).min(n - 1);
        let high = (low + 1).min(n - 1);
        positions[low] + (t - low as f64) * (positions[high] - positions[low])
    };
    let mut peaks: Vec<Peak> = local_maxima(&y)
        .into_iter()
        .filter(|&i| y[i] > min_height && (y[i] - y[i - 1]).min(y[i] - y[i + 1]) >= threshold)
        .map(|i| measure_peak(&y, i))
        .filter(|p| {
            let width = place(p.right) - place(p.left);
            p.prominence >= min_prominence && width >= min_width && width <= max_width
        })
        .collect();
    if min_distance > 0.0 {
        let mut order: Vec<usize> = (0..peaks.len()).collect();
        order.sort_by(|a, b| y[peaks[*b].index].total_cmp(&y[peaks[*a].index]));
        let mut removed = vec![false; peaks.len()];
        for &i in &order {
            if removed[i] {
                continue;
            }
            for &j in &order {
                if j != i
                    && !removed[j]
                    && (positions[peaks[j].index] - positions[peaks[i].index]).abs() <= min_distance
                {
                    removed[j] = true;
                }
            }
        }
        let mut keep = removed.into_iter();
        peaks.retain(|_| !keep.next().unwrap_or(false));
    }
    match sort.as_str() {
        "descend" => peaks.sort_by(|a, b| y[b.index].total_cmp(&y[a.index])),
        "ascend" => peaks.sort_by(|a, b| y[a.index].total_cmp(&y[b.index])),
        _ => {}
    }
    peaks.truncate(max_peaks);
    let orient = |values: Vec<f64>| {
        if m.rows == 1 && m.cols > 1 || (m.rows == 1 && values.is_empty()) {
            row(values)
        } else {
            col(values)
        }
    };
    let heights = peaks.iter().map(|p| y[p.index]).collect();
    let locations = peaks.iter().map(|p| positions[p.index]).collect();
    let widths = peaks
        .iter()
        .map(|p| place(p.right) - place(p.left))
        .collect();
    let prominences = peaks.iter().map(|p| p.prominence).collect();
    Ok(crate::spectral::outputs(
        session,
        vec![
            orient(heights),
            orient(locations),
            orient(widths),
            orient(prominences),
        ],
    ))
}

// medfilt1

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

fn medfilt1(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let m = matrix_arg(args, 0)?;
    if !m.is_real() {
        return Err(MatError::new("the input must be real"));
    }
    let mut truncate = false;
    let mut omit_nan = false;
    let mut numbers = Vec::new();
    for arg in &args[1..] {
        match arg {
            Value::Char(s) => match s.to_ascii_lowercase().as_str() {
                "zeropad" => truncate = false,
                "truncate" => truncate = true,
                "omitnan" => omit_nan = true,
                "includenan" => omit_nan = false,
                other => return Err(MatError::new(format!("unknown option '{other}'"))),
            },
            other => numbers.push(other.clone().into_matrix()?),
        }
    }
    let n = match numbers.first() {
        Some(v) if v.numel() > 0 => positive(v.real_scalar()?)?,
        _ => 3,
    };
    let dim = match numbers.get(2) {
        Some(v) if v.numel() > 0 => {
            let d = positive(v.real_scalar()?)?;
            if d != 1 && d != 2 {
                return Err(MatError::new("dimension must be 1 or 2"));
            }
            d
        }
        _ => default_dim(&m),
    };
    let len = length_along(&m, dim);
    Ok(Value::Matrix(along(&m, dim, len, |x| {
        let values: Vec<f64> = x.iter().map(|z| z.re).collect();
        (0..values.len())
            .map(|k| {
                let low = k as isize - (n / 2) as isize;
                let mut window = Vec::with_capacity(n);
                for t in low..low + n as isize {
                    if t >= 0 && (t as usize) < values.len() {
                        window.push(values[t as usize]);
                    } else if !truncate {
                        window.push(0.0);
                    }
                }
                if omit_nan {
                    window.retain(|v| !v.is_nan());
                } else if window.iter().any(|v| v.is_nan()) {
                    return Complex64::new(f64::NAN, 0.0);
                }
                Complex64::new(median(&mut window), 0.0)
            })
            .collect()
    })?))
}

// envelope

/// Magnitude of the analytic signal of `x`.
fn analytic_magnitude(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut buffer: Vec<Complex64> = x.iter().map(|v| Complex64::new(*v, 0.0)).collect();
    FftPlanner::new().plan_fft_forward(n).process(&mut buffer);
    for (k, v) in buffer.iter_mut().enumerate() {
        let weight = if k == 0 || (n.is_multiple_of(2) && k == n / 2) {
            1.0
        } else if k < n.div_ceil(2) {
            2.0
        } else {
            0.0
        };
        *v *= weight;
    }
    FftPlanner::new().plan_fft_inverse(n).process(&mut buffer);
    buffer.iter().map(|v| v.norm() / n as f64).collect()
}

fn moving_mean(x: &[f64], window: usize) -> Vec<f64> {
    let before = (window - 1) / 2;
    let after = window - before - 1;
    (0..x.len())
        .map(|k| {
            let low = k.saturating_sub(before);
            let high = (k + after).min(x.len() - 1);
            x[low..=high].iter().sum::<f64>() / (high - low + 1) as f64
        })
        .collect()
}

fn envelope(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let m = matrix_arg(args, 0)?;
    if !m.is_real() {
        return Err(MatError::new("the input must be real"));
    }
    let mut length = None;
    let mut method = "analytic".to_string();
    for arg in &args[1..] {
        match arg {
            Value::Char(s) => method = s.to_ascii_lowercase(),
            value => length = Some(positive(value.real_scalar()?)?),
        }
    }
    let dim = default_dim(&m);
    let len = length_along(&m, dim);
    let envelope_of = |x: &[Complex64]| -> MatResult<Vec<f64>> {
        let values: Vec<f64> = x.iter().map(|z| z.re).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let centered: Vec<f64> = values.iter().map(|v| v - mean).collect();
        match method.as_str() {
            "analytic" if length.is_none() => Ok(analytic_magnitude(&centered)),
            "analytic" => Err(MatError::new(
                "the analytic method with a filter length is not supported",
            )),
            "rms" => {
                let np =
                    length.ok_or_else(|| MatError::new("the rms method needs a window length"))?;
                let squares: Vec<f64> = centered.iter().map(|v| v * v).collect();
                Ok(moving_mean(&squares, np).iter().map(|v| v.sqrt()).collect())
            }
            other => Err(MatError::new(format!(
                "unsupported envelope method '{other}'"
            ))),
        }
    };
    let mut upper = Matrix::zeros(m.rows, m.cols);
    let mut lower = Matrix::zeros(m.rows, m.cols);
    let lines = if dim == 1 { m.cols } else { m.rows };
    for line in 0..lines {
        let x: Vec<Complex64> = (0..len)
            .map(|i| {
                if dim == 1 {
                    m.get(i, line)
                } else {
                    m.get(line, i)
                }
            })
            .collect();
        let mean = x.iter().map(|z| z.re).sum::<f64>() / len as f64;
        let magnitude = envelope_of(&x)?;
        for (i, v) in magnitude.iter().enumerate() {
            let (r, c) = if dim == 1 { (i, line) } else { (line, i) };
            upper.set(r, c, Complex64::new(mean + v, 0.0));
            lower.set(r, c, Complex64::new(mean - v, 0.0));
        }
    }
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![
            Value::Matrix(upper),
            Value::Matrix(lower),
        ]))
    } else {
        Ok(Value::Matrix(upper))
    }
}

#[cfg(test)]
mod tests {
    use crate::exec;

    const SETUP: &str = "xx = sin(2*pi*(0:99)'/20) + 0.5*cos(2*pi*(0:99)'/7); xi = sin(2*pi*(0:19)'/10) + 0.2*(0:19)'/20; v7 = [4 1 7 3 9 2 8 5 6]'; pk = [1 3 2 5 5 1 4 4 4 0 2 6 1]'; t = (0:63)'/64; e = (1+0.5*cos(2*pi*2*t)).*sin(2*pi*10*t) + 0.3;";

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

    /// Values from MATLAB R2025a (Signal Processing Toolbox 25.1).
    #[test]
    fn matches_matlab_reference() {
        let cases: &[(&str, &[f64])] = &[
            (
                r#"v = [upsample((1:6)', 2)' upsample(1:3, 3.0, 2)];"#,
                &[
                    1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 2.0, 0.0, 0.0, 3.0,
                ],
            ),
            (
                r#"v = [downsample((1:7)', 2)' downsample(1:9, 3.0, 1)];"#,
                &[1.0, 3.0, 5.0, 7.0, 2.0, 5.0, 8.0],
            ),
            (
                r#"y = upfirdn((1:6)', [1 2 1.0], 2.0, 3); v = [size(y) y'];"#,
                &[5.0, 1.0, 1.0, 4.0, 7.0, 10.0, 6.0],
            ),
            (
                r#"y = upfirdn(1:4, [1 -1 2.0], 3); v = [size(y) y];"#,
                &[
                    1.0, 12.0, 1.0, -1.0, 2.0, 2.0, -2.0, 4.0, 3.0, -3.0, 6.0, 4.0, -4.0, 8.0,
                ],
            ),
            (
                r#"y = upfirdn([1 2; 3 4; 5 6.0], [1 1.0], 2); v = [size(y) y(:)'];"#,
                &[
                    6.0, 2.0, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0,
                ],
            ),
            (
                r#"[y, b] = interp(xi, 3); bb = b(:)'; v = [size(y) size(b) y(1:7)' y(end-2:end)' bb(1:6) bb(13)];"#,
                &[
                    60.0,
                    1.0,
                    25.0,
                    1.0,
                    -1.8407087262042e-28,
                    0.211305757397374,
                    0.413469501819086,
                    0.597785252292482,
                    0.756642233570089,
                    0.882859349531646,
                    0.971056516295167,
                    -0.397785252292466,
                    -0.249677168309563,
                    -0.120071697414995,
                    -2.25318454711359e-13,
                    -0.00566051374920335,
                    -0.00616092081051452,
                    9.66773572061803e-13,
                    0.0323077982072248,
                    0.0366668811454165,
                    0.999999999995875,
                ],
            ),
            (
                r#"[y, b] = interp(xi, 2.0, 3.0, 0.4); v = [size(y) size(b) b(:)' y(1:6)'];"#,
                &[
                    40.0,
                    1.0,
                    13.0,
                    1.0,
                    -9.69696416773133e-13,
                    0.0184023091491313,
                    4.00501098642839e-12,
                    -0.11429209669581,
                    -8.11827502152875e-12,
                    0.596122395777611,
                    1.00000000001011,
                    0.596122395777611,
                    -8.11827502152875e-12,
                    -0.11429209669581,
                    4.00501098642839e-12,
                    0.0184023091491313,
                    -9.69696416773133e-13,
                    -2.01948391736579e-28,
                    0.313875244449443,
                    0.597785252292503,
                    0.823646776736455,
                    0.971056516295201,
                    1.02454539065633,
                ],
            ),
            (
                r#"y = interp(xi', 2); v = [size(y) y(1:5)];"#,
                &[
                    1.0,
                    40.0,
                    -4.20677946552909e-29,
                    0.314089894481077,
                    0.59778525229248,
                    0.824207308954224,
                    0.971056516295164,
                ],
            ),
            (
                r#"y = decimate(xx, 4); v = [numel(y) y(1:6)' y(end-2:end)'];"#,
                &[
                    25.0,
                    0.689985523647352,
                    0.812480812302677,
                    -0.281157031499356,
                    -1.03945330292793,
                    -0.273259381367442,
                    0.776408590318751,
                    -0.258441956154645,
                    -1.10437641872981,
                    0.00487051411118356,
                ],
            ),
            (
                r#"y = decimate(xx, 3); v = [numel(y) y(1:4)' y(end)];"#,
                &[
                    34.0,
                    0.46734327359679,
                    0.389704930965636,
                    1.29025126503899,
                    0.0636610488371526,
                    0.0810106993701032,
                ],
            ),
            (
                r#"y = decimate(xx, 3.0, 5); v = y(1:5)';"#,
                &[
                    0.520266196780468,
                    0.390564807617621,
                    1.25266016838183,
                    0.164081631536769,
                    -0.628533600104527,
                ],
            ),
            (
                r#"y = decimate(xx, 4.0, 'fir'); v = [numel(y) y(1:6)' y(end-2:end)'];"#,
                &[
                    25.0,
                    0.5,
                    0.765597288922496,
                    0.684267028794463,
                    -0.621996868771753,
                    -0.981516016613268,
                    0.0719237305240263,
                    0.482649923625401,
                    -0.495300703778865,
                    -1.07113127361153,
                ],
            ),
            (
                r#"y = decimate(xx, 3.0, 12.0, 'fir'); v = [numel(y) y(1:5)' y(end)];"#,
                &[
                    34.0,
                    0.5,
                    0.483064569746126,
                    1.13334578763629,
                    0.230585581502618,
                    -0.6477215789246,
                    0.00272790655441848,
                ],
            ),
            (
                r#"y = decimate(xx', 2); v = [size(y) y(1:4)];"#,
                &[
                    1.0,
                    50.0,
                    0.544604710452192,
                    0.401439200211472,
                    0.856119317820218,
                    1.32464547494672,
                ],
            ),
            (
                r#"[y, b] = resample(xx, 3.0, 2); v = [numel(y) size(b) y(1:5)' y(end-2:end)' b(1:3) b(31)];"#,
                &[
                    150.0,
                    1.0,
                    61.0,
                    0.500303086776889,
                    0.645952053392729,
                    0.573022784803285,
                    0.476813642036893,
                    0.386415316629052,
                    -0.0878384653908284,
                    -0.00318944418253101,
                    0.00100022458583095,
                    -1.43192112886976e-18,
                    -0.00152272743066009,
                    -0.0021514845240351,
                    1.00060617355378,
                ],
            ),
            (
                r#"y = resample(xx, 2.0, 5); v = [numel(y) y(1:5)' y(end-2:end)'];"#,
                &[
                    40.0,
                    0.420788264291645,
                    0.442111962784854,
                    0.86004031251183,
                    1.17788565278312,
                    -0.463183319552652,
                    -0.607637203046419,
                    -1.43366446276401,
                    -0.288818854515149,
                ],
            ),
            (
                r#"y = resample(xx, 3.0, 2.0, 5.0, 8); v = y(1:5)';"#,
                &[
                    0.500033308218216,
                    0.636262025955415,
                    0.582590918659864,
                    0.476556529697385,
                    0.377139842568039,
                ],
            ),
            (
                r#"y = resample(xx, 1.0, 3); v = [numel(y) y(1:5)'];"#,
                &[
                    34.0,
                    0.420997415329811,
                    0.421858788992781,
                    1.22266967218301,
                    0.222819529444571,
                    -0.712646538246863,
                ],
            ),
            (
                r#"y = resample(xx', 5.0, 4); v = [size(y) y(1:4)];"#,
                &[
                    1.0,
                    125.0,
                    0.500325258290718,
                    0.641171101565232,
                    0.533222924204134,
                    0.421022217803395,
                ],
            ),
            (
                r#"v = [medfilt1(v7)' medfilt1(v7, 4)' medfilt1(v7, 5)'];"#,
                &[
                    1.0, 4.0, 3.0, 7.0, 3.0, 8.0, 5.0, 6.0, 5.0, 0.5, 2.5, 3.5, 5.0, 5.0, 5.5, 6.5,
                    5.5, 5.5, 1.0, 3.0, 4.0, 3.0, 7.0, 5.0, 6.0, 5.0, 5.0,
                ],
            ),
            (
                r#"v = [medfilt1(v7, 3.0, [], 1.0, 'truncate')' medfilt1(v7, 4.0, [], 1.0, 'truncate')'];"#,
                &[
                    2.5, 4.0, 3.0, 7.0, 3.0, 8.0, 5.0, 6.0, 5.5, 2.5, 4.0, 3.5, 5.0, 5.0, 5.5, 6.5,
                    5.5, 6.0,
                ],
            ),
            (
                r#"a = medfilt1([1 NaN 3 4 2.0], 3.0, [], 2.0, 'omitnan'); v = a;"#,
                &[0.5, 2.0, 3.5, 3.0, 2.0],
            ),
            (
                r#"A = [4 1 7; 3 9 2; 8 5 6; 1 2 3.0]; y = medfilt1(A, 3); v = y(:)';"#,
                &[3.0, 4.0, 3.0, 1.0, 1.0, 5.0, 5.0, 2.0, 2.0, 6.0, 3.0, 3.0],
            ),
            (
                r#"[u, l] = envelope(e); v = [size(u) u(1:5)' l(1:5)' u(30:32)'];"#,
                &[
                    64.0,
                    1.0,
                    1.8,
                    1.79039264020162,
                    1.76193976625564,
                    1.71573480615127,
                    1.65355339059327,
                    -1.2,
                    -1.19039264020161,
                    -1.16193976625564,
                    -1.11573480615127,
                    -1.05355339059327,
                    1.71573480615127,
                    1.76193976625564,
                    1.79039264020162,
                ],
            ),
            (
                r#"[u, l] = envelope(e, 8.0, 'rms'); v = [u(1:5)' l(1:5)' u(30:32)'];"#,
                &[
                    1.23298089176984,
                    1.29355080877715,
                    1.23585027882477,
                    1.20156264619011,
                    1.26840859403731,
                    -0.632980891769843,
                    -0.693550808777154,
                    -0.635850278824772,
                    -0.601562646190108,
                    -0.668408594037305,
                    1.27892587679014,
                    1.37719475205314,
                    1.2866914445182,
                ],
            ),
            (
                r#"[p, l, w, pr] = findpeaks(pk); v = [size(p) p' l' w' pr'];"#,
                &[
                    4.0,
                    1.0,
                    3.0,
                    5.0,
                    4.0,
                    6.0,
                    2.0,
                    4.0,
                    7.0,
                    12.0,
                    0.75,
                    2.16666666666667,
                    2.875,
                    1.125,
                    1.0,
                    4.0,
                    3.0,
                    5.0,
                ],
            ),
            (
                r#"[p, l] = findpeaks(pk, 'MinPeakHeight', 3); v = [p' l'];"#,
                &[5.0, 4.0, 6.0, 4.0, 7.0, 12.0],
            ),
            (
                r#"[p, l] = findpeaks(pk, 'MinPeakDistance', 3); v = [p' l'];"#,
                &[5.0, 6.0, 4.0, 12.0],
            ),
            (
                r#"[p, l] = findpeaks(pk, 'Threshold', 1); v = [p' l'];"#,
                &[3.0, 6.0, 2.0, 12.0],
            ),
            (
                r#"[p, l] = findpeaks(pk, 'NPeaks', 2); v = [p' l'];"#,
                &[3.0, 5.0, 2.0, 4.0],
            ),
            (
                r#"[p, l] = findpeaks(pk, 'SortStr', 'descend'); v = [p' l'];"#,
                &[6.0, 5.0, 4.0, 3.0, 12.0, 4.0, 7.0, 2.0],
            ),
            (
                r#"[p, l] = findpeaks(pk, 'MinPeakProminence', 4); v = [p' l'];"#,
                &[5.0, 6.0, 4.0, 12.0],
            ),
            (
                r#"[p, l] = findpeaks(pk, 'MinPeakWidth', 1.5); v = [p' l'];"#,
                &[5.0, 4.0, 4.0, 7.0],
            ),
            (
                r#"[p, l, w] = findpeaks(pk', 'MaxPeakWidth', 2); v = [size(p) p l w];"#,
                &[1.0, 2.0, 3.0, 6.0, 2.0, 12.0, 0.75, 1.125],
            ),
            (
                r#"[p, l, w] = findpeaks(pk, 10); v = [p' l' w'];"#,
                &[
                    3.0,
                    5.0,
                    4.0,
                    6.0,
                    0.1,
                    0.3,
                    0.6,
                    1.1,
                    0.075,
                    0.216666666666667,
                    0.2875,
                    0.1125,
                ],
            ),
            (
                r#"[p, l] = findpeaks(pk, (1:13)'*2); v = [p' l'];"#,
                &[3.0, 5.0, 4.0, 6.0, 4.0, 8.0, 14.0, 24.0],
            ),
            (
                r#"ys = sin(2*pi*(0:99)'/20) + 0.4*sin(2*pi*(0:99)'/7); [p, l, w, pr] = findpeaks(ys); v = [numel(p) p(1:4)' l(1:4)' w(1:4)' pr(1:4)'];"#,
                &[
                    14.0,
                    0.982570490021971,
                    0.900517845279685,
                    -0.561085351422424,
                    1.19898815924768,
                    4.0,
                    9.0,
                    17.0,
                    24.0,
                    8.60513154403702,
                    2.03556573490148,
                    1.99806591151819,
                    10.1035861375029,
                    0.982570490021971,
                    0.290489010152415,
                    0.200253396517072,
                    2.32073774660984,
                ],
            ),
            (
                r#"[p, l] = findpeaks([1 1 1.0]'); v = [size(p) size(l)];"#,
                &[0.0, 1.0, 0.0, 1.0],
            ),
            (
                r#"X = abs(fft([1 2 3 4; 5 6 7 8.0], [], 2)); v = X(:)';"#,
                &[
                    10.0,
                    26.0,
                    2.82842712474619,
                    2.82842712474619,
                    2.0,
                    2.0,
                    2.82842712474619,
                    2.82842712474619,
                ],
            ),
            (
                r#"X = fft([1 2 3 4; 5 6 7 8.0], 3.0, 1); v = [size(X) real(X(:))' imag(X(:))'];"#,
                &[
                    3.0,
                    4.0,
                    6.0,
                    -1.5,
                    -1.5,
                    8.0,
                    -1.0,
                    -1.0,
                    10.0,
                    -0.5,
                    -0.5,
                    12.0,
                    0.0,
                    0.0,
                    0.0,
                    -4.33012701892219,
                    4.33012701892219,
                    0.0,
                    -5.19615242270663,
                    5.19615242270663,
                    0.0,
                    -6.06217782649107,
                    6.06217782649107,
                    0.0,
                    -6.92820323027551,
                    6.92820323027551,
                ],
            ),
            (
                r#"X = fft2([1 2 3; 4 5 6.0]); v = [size(X) real(X(:))' imag(X(:))'];"#,
                &[
                    2.0,
                    3.0,
                    21.0,
                    -9.0,
                    -3.0,
                    0.0,
                    -3.0,
                    0.0,
                    0.0,
                    0.0,
                    1.73205080756888,
                    0.0,
                    -1.73205080756888,
                    0.0,
                ],
            ),
            (
                r#"X = fft2(ones(2,3), 4.0, 5); v = [size(X) real(X(1,1)) imag(X(2,2))];"#,
                &[4.0, 5.0, 6.0, -2.03884176858763],
            ),
            (
                r#"Y = ifft2(fft2([8 1 6; 3 5 7; 4 9 2.0])); v = Y(:)';"#,
                &[8.0, 3.0, 4.0, 1.0, 5.0, 9.0, 6.0, 7.0, 2.0],
            ),
            (
                r#"x = [1 2 3 4.0]; y = ifft(fft(x)); v = y;"#,
                &[1.0, 2.0, 3.0, 4.0],
            ),
        ];
        for (code, want) in cases {
            let got = reals(&format!("{SETUP} {code}"), "v");
            assert_eq!(got.len(), want.len(), "{code}: {got:?}");
            for (g, w) in got.iter().zip(*want) {
                assert!(
                    (g - w).abs() <= 1e-7 * w.abs() + 1e-9,
                    "{code}: {got:?} vs {want:?}"
                );
            }
        }
    }

    #[test]
    fn rejects_bad_arguments() {
        for src in [
            "y = upsample([1 2 3], 2, 2);",
            "y = interp([1 2 3], 2);",
            "y = findpeaks([1 3 2], 'Bogus', 1);",
            "y = findpeaks([1 3 2], 'SortStr', 'up');",
            "y = decimate([1 2 3 4], 2, 'fir', 'x');",
            "y = medfilt1([1 2 3], 3, [], 3);",
        ] {
            assert!(exec(src).is_err(), "{src}");
        }
    }
}
