use num_complex::Complex64;
use openmat_core::{MatError, MatResult, Matrix, Series, SeriesKind, Session, System, Value};

use crate::numeric::complex_eigenvalues;
use crate::numeric::to_dmatrix;
use crate::util::{col, matrix_arg, need};

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("tf", tf as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("ss", ss),
        ("step", step),
        ("bode", bode),
        ("pole", pole),
    ] {
        session.register_builtin(name, func);
    }
}

fn tf(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    Ok(Value::System(System::Tf {
        num: matrix_arg(args, 0)?.reals()?,
        den: matrix_arg(args, 1)?.reals()?,
    }))
}

fn ss(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 4)?;
    Ok(Value::System(System::Ss {
        a: matrix_arg(args, 0)?,
        b: matrix_arg(args, 1)?,
        c: matrix_arg(args, 2)?,
        d: matrix_arg(args, 3)?,
    }))
}

fn step(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let (a, b, c, d) = to_ss(&args[0])?;
    let t_end = if args.len() >= 2 {
        args[1].real_scalar()?
    } else {
        10.0
    };
    let n = 201usize;
    let dt = t_end / (n - 1) as f64;
    let mut state = vec![0.0; a.rows];
    let mut ts = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let t = dt * i as f64;
        ts.push(t);
        ys.push(output(&c, &d, &state, 1.0));
        state = rk4(&a, &b, &state, 1.0, dt);
    }
    crate::plotb::prepare_plot_target(session);
    push_series(session, ts.clone(), ys.clone(), SeriesKind::Line);
    if session.nargout >= 2 {
        Ok(Value::Multi(vec![col(ts), col(ys)]))
    } else {
        Ok(col(ys))
    }
}

fn bode(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let (num, den) = tf_coeff(&args[0])?;
    let n = 80usize;
    let mut ws = Vec::new();
    let mut mag = Vec::new();
    let mut phase = Vec::new();
    for i in 0..n {
        let logw = -2.0 + 4.0 * i as f64 / (n - 1) as f64;
        let w = 10f64.powf(logw);
        let s = Complex64::new(0.0, w);
        let h = polyval(&num, s) / polyval(&den, s);
        ws.push(w);
        mag.push(20.0 * h.norm().max(1e-30).log10());
        phase.push(h.arg().to_degrees());
    }
    crate::plotb::prepare_plot_target(session);
    push_series(session, ws.clone(), mag.clone(), SeriesKind::Line);
    let axes = session.current_figure_mut().axes_mut();
    axes.series.push(Series {
        xs: ws,
        ys: phase,
        kind: SeriesKind::Line,
        label: "phase".into(),
    });
    Ok(crate::util::empty())
}

fn pole(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (num, den) = tf_coeff(&args[0])?;
    let _ = num;
    let companion = companion_matrix(&den)?;
    let dm = to_dmatrix(&companion)?;
    let values = complex_eigenvalues(&dm)?;
    let mut out = Matrix::zeros(values.len(), 1);
    for (i, v) in values.iter().enumerate() {
        out.set(i, 0, *v);
    }
    Ok(Value::Matrix(out))
}

fn tf_coeff(value: &Value) -> MatResult<(Vec<f64>, Vec<f64>)> {
    match value {
        Value::System(System::Tf { num, den }) => Ok((num.clone(), den.clone())),
        Value::System(System::Ss { .. }) => {
            Err(MatError::new("bode and pole currently take a tf system"))
        }
        _ => Err(MatError::new("expected a tf system")),
    }
}

fn to_ss(value: &Value) -> MatResult<(Matrix, Matrix, Matrix, Matrix)> {
    match value {
        Value::System(System::Ss { a, b, c, d }) => Ok((a.clone(), b.clone(), c.clone(), d.clone())),
        Value::System(System::Tf { num, den }) => tf_to_ss(num, den),
        _ => Err(MatError::new("expected a tf or ss system")),
    }
}

fn tf_to_ss(num: &[f64], den: &[f64]) -> MatResult<(Matrix, Matrix, Matrix, Matrix)> {
    if den.is_empty() || den[0] == 0.0 {
        return Err(MatError::new("tf denominator must be non-empty"));
    }
    let mut num: Vec<f64> = num.iter().map(|v| v / den[0]).collect();
    let den: Vec<f64> = den.iter().map(|v| v / den[0]).collect();
    while num.len() < den.len() {
        num.insert(0, 0.0);
    }
    if num.len() > den.len() {
        return Err(MatError::new("improper transfer functions are not supported"));
    }
    let d_gain = num[0];
    for (n, d) in num.iter_mut().zip(den.iter()) {
        *n -= d_gain * d;
    }
    let n = den.len() - 1;
    if n == 0 {
        return Ok((
            Matrix::zeros(1, 1),
            Matrix::zeros(1, 1),
            Matrix::zeros(1, 1),
            Matrix::scalar(d_gain),
        ));
    }
    let b: Vec<f64> = num[1..].to_vec();
    let mut a = Matrix::zeros(n, n);
    for col in 0..n {
        a.set(0, col, Complex64::from(-den[col + 1]));
    }
    for i in 1..n {
        a.set(i, i - 1, Complex64::from(1.0));
    }
    let mut bmat = Matrix::zeros(n, 1);
    bmat.set(0, 0, Complex64::from(1.0));
    let mut c = Matrix::zeros(1, n);
    for i in 0..n {
        c.set(0, i, Complex64::from(b.get(i).copied().unwrap_or(0.0)));
    }
    Ok((a, bmat, c, Matrix::scalar(d_gain)))
}

fn output(c: &Matrix, d: &Matrix, state: &[f64], u: f64) -> f64 {
    let mut y = if d.numel() == 0 { 0.0 } else { d.get(0, 0).re * u };
    for i in 0..state.len() {
        y += c.get(0, i).re * state[i];
    }
    y
}

fn deriv(a: &Matrix, b: &Matrix, state: &[f64], u: f64) -> Vec<f64> {
    let mut out = vec![0.0; state.len()];
    for r in 0..state.len() {
        let mut acc = b.get(r, 0).re * u;
        for c in 0..state.len() {
            acc += a.get(r, c).re * state[c];
        }
        out[r] = acc;
    }
    out
}

fn rk4(a: &Matrix, b: &Matrix, state: &[f64], u: f64, dt: f64) -> Vec<f64> {
    let k1 = deriv(a, b, state, u);
    let s2 = add_scaled(state, &k1, dt / 2.0);
    let k2 = deriv(a, b, &s2, u);
    let s3 = add_scaled(state, &k2, dt / 2.0);
    let k3 = deriv(a, b, &s3, u);
    let s4 = add_scaled(state, &k3, dt);
    let k4 = deriv(a, b, &s4, u);
    state
        .iter()
        .enumerate()
        .map(|(i, x)| x + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

fn add_scaled(state: &[f64], k: &[f64], scale: f64) -> Vec<f64> {
    state.iter().zip(k).map(|(s, dk)| s + scale * dk).collect()
}

fn companion_matrix(den: &[f64]) -> MatResult<Matrix> {
    if den.len() < 2 || den[0] == 0.0 {
        return Err(MatError::new("pole needs a denominator of degree at least 1"));
    }
    let n = den.len() - 1;
    let mut a = Matrix::zeros(n, n);
    for col in 0..n {
        a.set(0, col, Complex64::from(-den[col + 1] / den[0]));
    }
    for i in 1..n {
        a.set(i, i - 1, Complex64::from(1.0));
    }
    Ok(a)
}

fn polyval(coeff: &[f64], s: Complex64) -> Complex64 {
    let mut acc = Complex64::new(0.0, 0.0);
    for c in coeff {
        acc = acc * s + c;
    }
    acc
}

fn push_series(session: &mut Session, xs: Vec<f64>, ys: Vec<f64>, kind: SeriesKind) {
    let axes = session.current_figure_mut().axes_mut();
    if !axes.hold {
        axes.series.clear();
    }
    axes.series.push(Series {
        xs,
        ys,
        kind,
        label: String::new(),
    });
}

pub fn step_final(num: &[f64], den: &[f64]) -> MatResult<f64> {
    let (a, b, c, d) = tf_to_ss(num, den)?;
    let mut state = vec![0.0; a.rows];
    let dt = 0.02;
    let mut y = 0.0;
    for _ in 0..500 {
        y = output(&c, &d, &state, 1.0);
        state = rk4(&a, &b, &state, 1.0, dt);
    }
    Ok(y)
}
