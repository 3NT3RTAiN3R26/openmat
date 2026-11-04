//! `fir2`, Parks-McClellan (`firpm`), and the order estimators.

use std::f64::consts::PI;

use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Session, Value};
use rustfft::FftPlanner;

use crate::util::{col, matrix_arg, need, real_scalar, row};

pub fn fir2(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let order = real_scalar(args, 0)?;
    if order < 1.0 || order.fract() != 0.0 {
        return Err(MatError::new("fir2 order must be a positive integer"));
    }
    let mut n = order as usize;
    let f = matrix_arg(args, 1)?.reals()?;
    let m = matrix_arg(args, 2)?.reals()?;
    if f.len() != m.len() || f.len() < 2 {
        return Err(MatError::new("fir2 frequency and magnitude vectors must be the same length"));
    }
    if f[0].abs() > 1e-12 || (f[f.len() - 1] - 1.0).abs() > 1e-12 || f.windows(2).any(|w| w[1] < w[0]) {
        return Err(MatError::new("fir2 frequency must be nondecreasing, starting at 0 and ending at 1"));
    }
    let mut npt: Option<usize> = None;
    let mut lap: Option<usize> = None;
    let mut window: Option<Vec<f64>> = None;
    for arg in &args[3..] {
        match arg {
            Value::Char(_) => {
                return Err(MatError::new(
                    "Input NPT in fir2 must be numeric, instead it was char.",
                ));
            }
            other => {
                let values = other.clone().into_matrix()?.reals()?;
                if values.len() == 1 && window.is_none() && npt.is_none() {
                    npt = Some(values[0].round() as usize);
                } else if values.len() == 1 && window.is_none() && lap.is_none() {
                    lap = Some(values[0].round() as usize);
                } else if window.is_none() {
                    window = Some(values);
                } else {
                    return Err(MatError::new("fir2 received an unexpected argument"));
                }
            }
        }
    }
    // A passband at Nyquist needs an even order. The default window is built afterwards.
    if m[m.len() - 1].abs() > 1e-12 && n % 2 == 1 {
        n += 1;
        session.writeln(
            "Warning: Odd order symmetric FIR filters must have a gain of zero at the Nyquist frequency. The order is being increased by one.",
        );
    }
    let mut npt = npt.unwrap_or(if n + 1 < 1024 { 512 } else { n + 1 });
    if !npt.is_power_of_two() {
        npt = npt.next_power_of_two();
    }
    if 2 * npt < n + 1 {
        return Err(MatError::new("fir2 grid size must be greater than half the filter order"));
    }
    let lap = lap.unwrap_or(npt / 25) as i32;
    let window = match window {
        Some(w) if w.len() == n + 1 => w,
        Some(_) => return Err(MatError::new("fir2 window must have order + 1 elements")),
        None => hamming(n + 1),
    };
    let mut freq = f.clone();
    let mag = m.clone();
    freq[0] = 0.0;
    let last = freq.len() - 1;
    freq[last] = 1.0;
    let npt_i = npt as i32 + 1;
    let mut hgrid = vec![0.0; npt_i as usize];
    hgrid[0] = mag[0];
    let mut nb = 1i32;
    for i in 0..freq.len() - 1 {
        let df = freq[i + 1] - freq[i];
        let ne = if df == 0.0 {
            nb = (nb as f64 - lap as f64 / 2.0).ceil() as i32;
            nb + lap
        } else {
            (freq[i + 1] * npt_i as f64).floor() as i32
        };
        if nb < 1 || ne > npt_i {
            return Err(MatError::new("fir2 could not place the transition grid"));
        }
        let span = (ne - nb) as f64;
        for j in nb..=ne {
            let inc = if nb == ne { 0.0 } else { (j - nb) as f64 / span };
            hgrid[(j - 1) as usize] = inc * mag[i + 1] + (1.0 - inc) * mag[i];
        }
        nb = ne + 1;
    }
    let dt = 0.5 * n as f64;
    let mut spectrum: Vec<Complex64> = hgrid
        .iter()
        .enumerate()
        .map(|(k, h)| {
            let rad = -dt * PI * k as f64 / (npt_i as f64 - 1.0);
            Complex64::from_polar(*h, rad)
        })
        .collect();
    let tail: Vec<Complex64> = spectrum[1..spectrum.len() - 1].iter().rev().map(|z| z.conj()).collect();
    spectrum.extend(tail);
    let len = spectrum.len();
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(len);
    ifft.process(&mut spectrum);
    let mut b: Vec<f64> = spectrum.iter().take(n + 1).map(|z| z.re / len as f64 * 1.0).collect();
    for (coef, w) in b.iter_mut().zip(&window) {
        *coef *= w;
    }
    Ok(row(b))
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

pub fn firpm(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let order = real_scalar(args, 0)?;
    if order < 3.0 || order.fract() != 0.0 {
        return Err(MatError::new("firpm order must be an integer of at least 3"));
    }
    let edges = matrix_arg(args, 1)?.reals()?;
    let amps = matrix_arg(args, 2)?.reals()?;
    if edges.len() != amps.len() || edges.len() < 2 || edges.len() % 2 != 0 {
        return Err(MatError::new("firpm needs an even number of band edges and one amplitude each"));
    }
    let mut weights = None;
    let mut kind = "bandpass";
    for arg in &args[3..] {
        match arg {
            Value::Char(text) => {
                kind = match text.to_ascii_lowercase().as_str() {
                    "bandpass" => "bandpass",
                    "differentiator" => "differentiator",
                    "hilbert" => "hilbert",
                    other => return Err(MatError::new(format!("firpm does not know the type '{other}'"))),
                };
            }
            other => weights = Some(other.clone().into_matrix()?.reals()?),
        }
    }
    let (h, err) = remez(order as usize + 1, &edges, &amps, weights.as_deref(), kind)?;
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![row(h), Value::scalar(err)]))
    } else {
        Ok(row(h))
    }
}

pub fn firpmord(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let edges = matrix_arg(args, 0)?.reals()?;
    let amps = matrix_arg(args, 1)?.reals()?;
    let dev = matrix_arg(args, 2)?.reals()?;
    let fs = if args.len() >= 4 { real_scalar(args, 3)? } else { 2.0 };
    if edges.len() < 2 || edges.len() % 2 != 0 || amps.len() != edges.len() / 2 + 1 && amps.len() != dev.len() {
        // MATLAB: f lists the transition edges, a and dev list one value per band.
    }
    if amps.len() != dev.len() || amps.len() != edges.len() / 2 + 1 {
        return Err(MatError::new("firpmord amplitudes and deviations need one entry per band"));
    }
    if fs <= 0.0 || dev.iter().any(|d| *d <= 0.0) {
        return Err(MatError::new("firpmord sample rate and deviations must be positive"));
    }
    let mut df = f64::MAX;
    for pair in edges.chunks(2) {
        df = df.min(pair[1] - pair[0]);
    }
    if df <= 0.0 {
        return Err(MatError::new("firpmord transition width must be positive"));
    }
    let dp = dev[0];
    let ds = *dev.last().unwrap();
    let n = ((2.0 / 3.0) * (1.0 / (10.0 * dp * ds)).log10() * fs / df - 1.0).ceil();
    let nyquist = fs / 2.0;
    let mut fo = vec![0.0];
    fo.extend(edges.iter().map(|f| f / nyquist));
    fo.push(1.0);
    let mut ao = Vec::new();
    for amp in &amps {
        ao.push(*amp);
        ao.push(*amp);
    }
    let dev_max = dev.iter().cloned().fold(0.0, f64::max);
    let w: Vec<f64> = dev.iter().map(|d| dev_max / d).collect();
    Ok(Value::Multi(vec![
        Value::scalar(n),
        col(fo),
        col(ao),
        col(w),
    ]))
}

pub fn kaiserord(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let freq = matrix_arg(args, 0)?.reals()?;
    let mag = matrix_arg(args, 1)?.reals()?;
    let dev = matrix_arg(args, 2)?.reals()?;
    let fs = if args.len() >= 4 { real_scalar(args, 3)? } else { 2.0 };
    if freq.len() != mag.len() * 2 - 2 || mag.len() != dev.len() || mag.is_empty() {
        return Err(MatError::new("kaiserord frequencies, magnitudes and deviations do not match"));
    }
    if fs <= 0.0 {
        return Err(MatError::new("kaiserord sample rate must be positive"));
    }
    let mut width = f64::MAX;
    let mut wn = Vec::new();
    for pair in freq.chunks(2) {
        width = width.min(pair[1] - pair[0]);
        wn.push((pair[0] + pair[1]) / fs);
    }
    let dev_min = dev.iter().cloned().fold(f64::MAX, f64::min);
    let a = -20.0 * dev_min.log10();
    let beta = if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    };
    let dw = PI2_OVER * width / fs;
    let mut n = ((a - 8.0) / (2.285 * dw)).ceil();
    let ftype = if mag.len() == 1 {
        "low"
    } else if wn.len() == 1 {
        if mag[0] > mag[1] { "low" } else { "high" }
    } else if wn.len() == 2 {
        if mag[0] > mag[1] { "stop" } else { "DC-0" }
    } else if mag[0] > mag[1] {
        "DC-1"
    } else {
        "DC-0"
    };
    let even_len = wn.len() % 2 == 0;
    let first_bigger = mag[0] > mag[1];
    if (first_bigger == even_len) && (n as i64) % 2 == 1 {
        n += 1.0;
    }
    Ok(Value::Multi(vec![
        Value::scalar(n),
        row(wn),
        Value::scalar(beta),
        Value::Char(ftype.to_string()),
    ]))
}

const PI2_OVER: f64 = 2.0 * PI;

pub fn firceqrip(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let order = real_scalar(args, 0)?;
    let fo = real_scalar(args, 1)?;
    let dev = matrix_arg(args, 2)?.reals()?;
    if dev.len() != 2 || dev[0] <= 0.0 || dev[1] <= 0.0 {
        return Err(MatError::new("firceqrip deviations must be a positive pair"));
    }
    if !(0.0..1.0).contains(&fo) || order < 3.0 {
        return Err(MatError::new("firceqrip needs an order of at least 3 and 0 < Fo < 1"));
    }
    let mut high = false;
    let mut slope = 0.0;
    let mut index = 3;
    while index < args.len() {
        match &args[index] {
            Value::Char(text) if text.eq_ignore_ascii_case("high") => high = true,
            Value::Char(text) if text.eq_ignore_ascii_case("slope") => {
                slope = args
                    .get(index + 1)
                    .ok_or_else(|| MatError::new("firceqrip slope needs a value"))?
                    .real_scalar()?;
                index += 1;
            }
            Value::Char(text) => {
                return Err(MatError::new(format!("firceqrip does not know the option '{text}'")));
            }
            _ => return Err(MatError::new("firceqrip options must be strings")),
        }
        index += 1;
    }
    let (pass_dev, stop_dev) = if high { (dev[1], dev[0]) } else { (dev[0], dev[1]) };
    // The transition is centered on Fo. Its width is the one that makes the weighted error 1,
    // which is the passband and stopband ripples asked for.
    let mut low = 1e-4;
    let mut high_w = (fo.min(1.0 - fo) - 1e-4).max(low);
    let mut best = None;
    for _ in 0..48 {
        let dw = (low + high_w) / 2.0;
        let edges = [0.0, fo - dw, fo + dw, 1.0];
        let amps = if high { [0.0, 0.0, 1.0, 1.0] } else { [1.0, 1.0, 0.0, 0.0] };
        let weights = [1.0 / pass_dev, 1.0 / stop_dev / (1.0 + slope * fo).max(1e-6)];
        match remez(order as usize + 1, &edges, &amps, Some(&weights), "bandpass") {
            Ok((h, err)) => {
                best = Some(h);
                if err > 1.0 {
                    low = dw;
                } else {
                    high_w = dw;
                }
            }
            Err(_) => low = dw,
        }
    }
    let h = best.ok_or_else(|| MatError::new("firceqrip did not converge"))?;
    let _ = session;
    Ok(row(h))
}

/// Parks-McClellan. `edges` and `amps` use MATLAB's 0..1 Nyquist normalization.
fn remez(numtaps: usize, edges: &[f64], amps: &[f64], weights: Option<&[f64]>, kind: &str) -> MatResult<(Vec<f64>, f64)> {
    let bands_n = edges.len() / 2;
    let weight = match weights {
        Some(w) if w.len() == bands_n => w.to_vec(),
        Some(_) => return Err(MatError::new("firpm needs one weight per band")),
        None => vec![1.0; bands_n],
    };
    let symmetry_positive = kind == "bandpass";
    let mut r = numtaps / 2;
    if numtaps % 2 == 1 && symmetry_positive {
        r += 1;
    }
    let bands: Vec<f64> = edges.iter().map(|f| f / 2.0).collect();
    let delf = 0.5 / (16.0 * r as f64);
    let grid0 = if !symmetry_positive && delf > bands[0] { delf } else { bands[0] };
    let mut grid = Vec::new();
    let mut desired = Vec::new();
    let mut weight_g = Vec::new();
    for band in 0..bands_n {
        let mut low = if band == 0 { grid0 } else { bands[2 * band] };
        let high = bands[2 * band + 1];
        let mut k = ((high - low) / delf + 0.5) as i32;
        if band == 0 && !symmetry_positive && numtaps % 2 == 0 {
            k -= 1;
        }
        if k < 2 {
            return Err(MatError::new("firpm band is too narrow for the frequency grid"));
        }
        let k = k as usize;
        for i in 0..k {
            let t = i as f64 / (k - 1) as f64;
            desired.push(amps[2 * band] + t * (amps[2 * band + 1] - amps[2 * band]));
            weight_g.push(weight[band]);
            grid.push(low);
            low += delf;
        }
        *grid.last_mut().unwrap() = high;
    }
    if !symmetry_positive && *grid.last().unwrap() > 0.5 - delf && numtaps % 2 == 1 {
        *grid.last_mut().unwrap() = 0.5 - delf;
    }
    if kind == "differentiator" {
        for i in 0..grid.len() {
            if desired[i] > 0.0001 {
                weight_g[i] = weight[band_of(i, &grid, &bands)] / grid[i].max(1e-8);
            }
        }
    }
    if symmetry_positive {
        if numtaps % 2 == 0 {
            for i in 0..grid.len() {
                let c = (PI * grid[i]).cos();
                desired[i] /= c;
                weight_g[i] *= c;
            }
        }
    } else if numtaps % 2 == 1 {
        for i in 0..grid.len() {
            let c = (2.0 * PI * grid[i]).sin();
            if c.abs() < 1e-12 {
                return Err(MatError::new("firpm hilbert grid hit a zero of the basis"));
            }
            desired[i] /= c;
            weight_g[i] *= c;
        }
    } else {
        for i in 0..grid.len() {
            let c = (PI * grid[i]).sin();
            if c.abs() < 1e-12 {
                return Err(MatError::new("firpm differentiator grid hit a zero of the basis"));
            }
            desired[i] /= c;
            weight_g[i] *= c;
        }
    }
    let mut ext: Vec<usize> = (0..=r).map(|i| i * (grid.len() - 1) / r).collect();
    let mut ad = vec![0.0; r + 1];
    let mut x = vec![0.0; r + 1];
    let mut y = vec![0.0; r + 1];
    let mut converged = false;
    for _ in 0..40 {
        calc_parms(r, &ext, &grid, &desired, &weight_g, &mut ad, &mut x, &mut y);
        let err = error_grid(r, &ad, &x, &y, &grid, &desired, &weight_g);
        ext = search(r, &err).ok_or_else(|| MatError::new("firpm failed to find enough extrema"))?;
        let values: Vec<f64> = ext.iter().map(|i| err[*i].abs()).collect();
        let max = values.iter().cloned().fold(0.0, f64::max);
        let min = values.iter().cloned().fold(f64::MAX, f64::min);
        if max > 0.0 && (max - min) / max < 1e-4 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(MatError::new("firpm did not converge"));
    }
    calc_parms(r, &ext, &grid, &desired, &weight_g, &mut ad, &mut x, &mut y);
    let mut taps = vec![0.0; r + 1];
    for i in 0..=numtaps / 2 {
        let c = if symmetry_positive {
            if numtaps % 2 == 1 { 1.0 } else { (PI * i as f64 / numtaps as f64).cos() }
        } else if numtaps % 2 == 1 {
            (2.0 * PI * i as f64 / numtaps as f64).sin()
        } else {
            (PI * i as f64 / numtaps as f64).sin()
        };
        taps[i] = compute_a(i as f64 / numtaps as f64, r, &ad, &x, &y) * c;
    }
    let h = freq_sample(numtaps, &taps, symmetry_positive);
    let ripple = {
        let err = error_grid(r, &ad, &x, &y, &grid, &desired, &weight_g);
        ext.iter().map(|i| err[*i].abs()).fold(0.0, f64::max)
    };
    Ok((h, ripple))
}

fn band_of(index: usize, grid: &[f64], bands: &[f64]) -> usize {
    let f = grid[index];
    for b in 0..bands.len() / 2 {
        if f <= bands[2 * b + 1] + 1e-12 {
            return b;
        }
    }
    bands.len() / 2 - 1
}

fn calc_parms(
    r: usize,
    ext: &[usize],
    grid: &[f64],
    desired: &[f64],
    weight: &[f64],
    ad: &mut [f64],
    x: &mut [f64],
    y: &mut [f64],
) {
    for i in 0..=r {
        x[i] = (2.0 * PI * grid[ext[i]]).cos();
    }
    let ld = (r - 1) / 15 + 1;
    for i in 0..=r {
        let mut denom = 1.0;
        let xi = x[i];
        for j in 0..ld {
            let mut k = j;
            while k <= r {
                if k != i {
                    denom *= 2.0 * (xi - x[k]);
                }
                k += ld;
            }
        }
        if denom.abs() < 1e-5 {
            denom = 1e-5;
        }
        ad[i] = 1.0 / denom;
    }
    let mut numer = 0.0;
    let mut denom = 0.0;
    let mut sign = 1.0;
    for i in 0..=r {
        numer += ad[i] * desired[ext[i]];
        denom += sign * ad[i] / weight[ext[i]];
        sign = -sign;
    }
    let delta = numer / denom;
    sign = 1.0;
    for i in 0..=r {
        y[i] = desired[ext[i]] - sign * delta / weight[ext[i]];
        sign = -sign;
    }
}

fn compute_a(freq: f64, r: usize, ad: &[f64], x: &[f64], y: &[f64]) -> f64 {
    let xc = (2.0 * PI * freq).cos();
    let mut numer = 0.0;
    let mut denom = 0.0;
    for i in 0..=r {
        let mut c = xc - x[i];
        if c.abs() < 1e-7 {
            return y[i];
        }
        c = ad[i] / c;
        denom += c;
        numer += c * y[i];
    }
    numer / denom
}

fn error_grid(r: usize, ad: &[f64], x: &[f64], y: &[f64], grid: &[f64], desired: &[f64], weight: &[f64]) -> Vec<f64> {
    grid.iter()
        .enumerate()
        .map(|(i, f)| weight[i] * (desired[i] - compute_a(*f, r, ad, x, y)))
        .collect()
}

fn search(r: usize, err: &[f64]) -> Option<Vec<usize>> {
    let mut found = Vec::new();
    if (err[0] > 0.0 && err[0] > err[1]) || (err[0] < 0.0 && err[0] < err[1]) {
        found.push(0);
    }
    for i in 1..err.len() - 1 {
        if (err[i] >= err[i - 1] && err[i] > err[i + 1] && err[i] > 0.0)
            || (err[i] <= err[i - 1] && err[i] < err[i + 1] && err[i] < 0.0)
        {
            found.push(i);
        }
    }
    let j = err.len() - 1;
    if (err[j] > 0.0 && err[j] > err[j - 1]) || (err[j] < 0.0 && err[j] < err[j - 1]) {
        found.push(j);
    }
    if found.len() < r + 1 {
        return None;
    }
    let mut extra = found.len() - (r + 1);
    while extra > 0 {
        let mut up = err[found[0]] > 0.0;
        let mut smallest = 0;
        let mut alternating = true;
        for j in 1..found.len() {
            if err[found[j]].abs() < err[found[smallest]].abs() {
                smallest = j;
            }
            if up && err[found[j]] < 0.0 {
                up = false;
            } else if !up && err[found[j]] > 0.0 {
                up = true;
            } else {
                alternating = false;
                break;
            }
        }
        if alternating && extra == 1 {
            smallest = if err[found[found.len() - 1]].abs() < err[found[0]].abs() {
                found.len() - 1
            } else {
                0
            };
        }
        found.remove(smallest);
        extra -= 1;
    }
    Some(found)
}

fn freq_sample(n: usize, a: &[f64], positive: bool) -> Vec<f64> {
    let m = (n - 1) as f64 / 2.0;
    let mut h = vec![0.0; n];
    if positive {
        let last = if n % 2 == 1 { m as usize } else { n / 2 - 1 };
        for sample in 0..n {
            let mut val = a[0];
            let x = 2.0 * PI * (sample as f64 - m) / n as f64;
            for k in 1..=last {
                val += 2.0 * a[k] * (x * k as f64).cos();
            }
            h[sample] = val / n as f64;
        }
    } else if n % 2 == 1 {
        // Odd antisymmetric (type III) filters, including a differentiator of even order.
        // The sine series above is the negative of the DFT sign MATLAB uses.
        for sample in 0..n {
            let mut val = 0.0;
            let x = 2.0 * PI * (sample as f64 - m) / n as f64;
            for k in 1..=m as usize {
                val += 2.0 * a[k] * (x * k as f64).sin();
            }
            h[sample] = -val / n as f64;
        }
    } else {
        for sample in 0..n {
            let mut val = a[n / 2] * (PI * (sample as f64 - m)).sin();
            let x = 2.0 * PI * (sample as f64 - m) / n as f64;
            for k in 1..n / 2 {
                val += 2.0 * a[k] * (x * k as f64).sin();
            }
            h[sample] = val / n as f64;
        }
    }
    h
}
