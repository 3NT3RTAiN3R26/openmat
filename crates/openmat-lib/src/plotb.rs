use openmat_core::{MatResult, Series, SeriesKind, Session, Value};

use crate::util::{char_arg, need, positive, real_vec};

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("plot", plot as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("stem", stem),
        ("figure", figure),
        ("clf", clf),
        ("hold", hold),
        ("xlabel", xlabel),
        ("ylabel", ylabel),
        ("title", title),
        ("subplot", subplot),
    ] {
        session.register_builtin(name, func);
    }
}

fn plot(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    add_series(session, args, SeriesKind::Line)
}

fn stem(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    add_series(session, args, SeriesKind::Stem)
}

pub(crate) fn prepare_plot_target(session: &mut Session) {
    if !session.new_figure_on_plot {
        return;
    }
    let index = session.current_figure.min(session.figures.len().saturating_sub(1));
    let Some(figure) = session.figures.get(index) else {
        return;
    };
    let active = figure.active.min(figure.axes.len().saturating_sub(1));
    let held = figure.axes.get(active).is_some_and(|axes| axes.hold);
    let busy = figure.axes.iter().any(|axes| !axes.series.is_empty());
    if busy && !held {
        session.figures.push(openmat_core::Figure::new());
        session.current_figure = session.figures.len() - 1;
    }
}

fn add_series(session: &mut Session, args: &[Value], kind: SeriesKind) -> MatResult<Value> {
    prepare_plot_target(session);
    need(args, 1)?;
    let (xs, ys) = if args.len() == 1 || matches!(args.get(1), Some(Value::Char(_))) {
        let ys = real_vec(&args[0])?;
        let xs = (1..=ys.len()).map(|i| i as f64).collect();
        (xs, ys)
    } else {
        (real_vec(&args[0])?, real_vec(&args[1])?)
    };
    let n = xs.len().min(ys.len());
    let axes = session.current_figure_mut().axes_mut();
    if !axes.hold {
        axes.series.clear();
    }
    axes.series.push(Series {
        xs: xs[..n].to_vec(),
        ys: ys[..n].to_vec(),
        kind,
        label: String::new(),
    });
    Ok(crate::util::empty())
}

fn figure(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let n = if args.is_empty() {
        session.figures.len() + 1
    } else {
        positive(args[0].real_scalar()?)?
    };
    if n == 0 {
        return Err(openmat_core::MatError::new("figure number starts at 1"));
    }
    while session.figures.len() < n {
        session.figures.push(openmat_core::Figure::new());
    }
    session.current_figure = n - 1;
    Ok(Value::scalar(n as f64))
}

fn clf(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    if let Some(arg) = args.first() {
        match arg {
            Value::Char(text) if text == "reset" => {}
            Value::Char(text) => {
                return Err(openmat_core::MatError::new(format!(
                    "clf does not support '{text}'"
                )));
            }
            _ => return Err(openmat_core::MatError::new("clf expects 'reset' or no arguments")),
        }
    }
    *session.current_figure_mut() = openmat_core::Figure::new();
    Ok(crate::util::empty())
}

fn hold(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let mode = if args.is_empty() {
        "toggle".to_string()
    } else {
        match &args[0] {
            Value::Char(s) => s.clone(),
            other => {
                if other.real_scalar().unwrap_or(0.0) != 0.0 {
                    "on".into()
                } else {
                    "off".into()
                }
            }
        }
    };
    let axes = session.current_figure_mut().axes_mut();
    axes.hold = match mode.as_str() {
        "on" => true,
        "off" => false,
        _ => !axes.hold,
    };
    Ok(crate::util::empty())
}

fn xlabel(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let text = label_text(args)?;
    session.current_figure_mut().axes_mut().xlabel = text;
    Ok(crate::util::empty())
}

fn ylabel(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let text = label_text(args)?;
    session.current_figure_mut().axes_mut().ylabel = text;
    Ok(crate::util::empty())
}

fn title(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let text = label_text(args)?;
    session.current_figure_mut().axes_mut().title = text;
    Ok(crate::util::empty())
}

fn label_text(args: &[Value]) -> MatResult<String> {
    need(args, 1)?;
    if let Ok(text) = char_arg(args, 0) {
        return Ok(text);
    }
    Ok(args[0].to_string())
}

fn subplot(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let rows = positive(args[0].real_scalar()?)?.max(1);
    let cols = positive(args[1].real_scalar()?)?.max(1);
    let index = positive(args[2].real_scalar()?)?;
    if index == 0 || index > rows * cols {
        return Err(openmat_core::MatError::new("subplot index is out of range"));
    }
    let figure = session.current_figure_mut();
    figure.rows = rows;
    figure.cols = cols;
    let need = rows * cols;
    while figure.axes.len() < need {
        figure.axes.push(openmat_core::Axes::new());
    }
    figure.active = index - 1;
    Ok(crate::util::empty())
}
