mod control;
mod dsp;
mod fixed;
mod hilb;
mod numeric;
mod plotb;
mod text;
mod util;

pub use openmat_core::*;

pub fn session() -> Session {
    let mut session = Session::new();
    register(&mut session);
    session
}

pub fn register(session: &mut Session) {
    numeric::register(session);
    dsp::register(session);
    control::register(session);
    plotb::register(session);
    fixed::register(session);
    text::register(session);
    hilb::register(session);
}

pub fn exec(src: &str) -> MatResult<Session> {
    let mut session = session();
    session.eval_source(src)?;
    Ok(session)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ans(src: &str) -> Value {
        let session = exec(src).unwrap_or_else(|err| panic!("{src}: {err}"));
        session.get("ans").cloned().unwrap_or_else(|| panic!("no ans for {src}"))
    }

    fn var(src: &str, name: &str) -> Value {
        let session = exec(src).unwrap();
        session.get(name).cloned().unwrap()
    }

    fn scalar(src: &str) -> f64 {
        ans(src).real_scalar().unwrap()
    }

    #[test]
    fn new_figure_option_opens_a_window_per_plot() {
        let mut session = session();
        session.new_figure_on_plot = true;
        session.eval_source("plot([1 2]); plot([3 4]);").unwrap();
        assert_eq!(session.figures.len(), 2);
        assert_eq!(session.figures[0].axes[0].series.len(), 1);
        assert_eq!(session.figures[1].axes[0].series.len(), 1);

        let session = exec("figure(1); plot([1 2]); figure(2); plot([3 4]);").unwrap();
        assert_eq!(session.figures.len(), 2);
        assert_eq!(session.current_figure, 1);
    }

    #[test]
    fn default_figure_window_style_applies_to_new_figures() {
        let session = exec(
            "plot([1 2]); set(0, 'DefaultFigureWindowStyle', 'docked'); figure; plot([3 4]);",
        )
        .unwrap();
        assert_eq!(
            session.default_figure_window_style,
            openmat_core::FigureWindowStyle::Docked
        );
        assert_eq!(
            session.figures[0].window_style,
            openmat_core::FigureWindowStyle::Normal
        );
        assert_eq!(
            session.figures[1].window_style,
            openmat_core::FigureWindowStyle::Docked
        );

        let session = exec("set(0, 'DefaultFigureWindowStyle', 'normal');").unwrap();
        assert_eq!(
            session.default_figure_window_style,
            openmat_core::FigureWindowStyle::Normal
        );
    }

    #[test]
    fn close_removes_one_figure_window() {
        let session = exec("figure(1); plot([1 2]); figure(2); plot([3 4]); close(1);").unwrap();
        assert!(session.figures[0].axes[0].series.is_empty());
        assert_eq!(session.figures[1].axes[0].series.len(), 1);
    }

    #[test]
    fn clf_clears_the_current_figure() {
        let session = exec("plot([1 2], [3 4]); title('sine'); clf;").unwrap();
        let axes = &session.figures[0].axes[0];
        assert!(axes.series.is_empty());
        assert!(axes.title.is_empty());
    }

    #[test]
    fn clear_removes_variables_and_home_clears_output() {
        let session = exec("x = 4; clear x;").unwrap();
        assert!(session.get("x").is_none());
        assert!(session.get("pi").is_some());

        let session = exec("a = 1; b = 2; clear;").unwrap();
        assert!(session.get("a").is_none());
        assert!(session.get("b").is_none());
        assert!((session.get("pi").unwrap().real_scalar().unwrap() - std::f64::consts::PI).abs() < 1e-12);

        let session = exec("fprintf('hi'); home; fprintf('yo');").unwrap();
        assert_eq!(session.output, "yo");
        assert!(session.output_homed);
    }

    #[test]
    fn arithmetic_and_matrix_literals() {
        assert!((scalar("[1 -2; 3 4]; ans(1,2)") - -2.0).abs() < 1e-9);
        assert!((scalar("[1 - 2]") - -1.0).abs() < 1e-9);
        assert!((scalar("[1-2]") - -1.0).abs() < 1e-9);
        assert!((scalar("a = [1 2 3]; a(end)") - 3.0).abs() < 1e-9);
        assert!((scalar("s = 0; for i = 1:5, s = s + i; end; s") - 15.0).abs() < 1e-9);
    }

    #[test]
    fn functions_and_anonymous() {
        assert!((scalar("function y = add(a, b)\ny = a + b;\nend\nadd(2, 3)") - 5.0).abs() < 1e-9);
        assert!((scalar("f = @(x) x * x; f(4)") - 16.0).abs() < 1e-9);
    }

    #[test]
    fn linear_algebra() {
        assert!((scalar("det([1 2; 3 4])") - -2.0).abs() < 1e-8);
        let session = exec("A = [2 0; 0 3]; e = eig(A);").unwrap();
        let e = session.get("e").unwrap().clone().into_matrix().unwrap().reals().unwrap();
        let mut e = e;
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((e[0] - 2.0).abs() < 1e-6);
        assert!((e[1] - 3.0).abs() < 1e-6);
        assert!((scalar("A = [1 2; 3 4]; b = [5; 11]; x = A \\ b; x(1)") - 1.0).abs() < 1e-6);
        let x2 = var("A = [1 2; 3 4]; b = [5; 11]; x = A \\ b; x", "x");
        let m = x2.into_matrix().unwrap();
        assert!((m.get(1, 0).re - 2.0).abs() < 1e-6);
    }

    #[test]
    fn control_flow() {
        assert_eq!(scalar("x = 2; if x > 1, y = 3; else, y = 4; end; y"), 3.0);
        assert_eq!(
            scalar("s = 0; switch 2, case 1, s = 1; case 2, s = 5; otherwise, s = 9; end; s"),
            5.0
        );
    }

    #[test]
    fn dsp_matches_known_values() {
        let fft = ans("fft([1 0 0 0])");
        let m = fft.into_matrix().unwrap();
        assert!((m.get(0, 0).re - 1.0).abs() < 1e-8);
        assert!((m.get(0, 1).re - 1.0).abs() < 1e-8);
        assert!((scalar("y = ifft(fft([1 2 3 4])); y(1)") - 1.0).abs() < 1e-6);
        assert!((scalar("conv([1 2], [3 4])(2)") - 10.0).abs() < 1e-8);
        let (b, a) = dsp::butterworth(1, 0.5).unwrap();
        assert!((b[0] - 0.5).abs() < 1e-6, "{b:?}");
        assert!((a[0] - 1.0).abs() < 1e-6);
        assert!(a[1].abs() < 1e-6, "{a:?}");
        let y = control::step_final(&[1.0], &[1.0, 1.0]).unwrap();
        assert!((y - 1.0).abs() < 0.05, "{y}");
    }

    #[test]
    fn fixed_point_hex() {
        let text = ans("fi2hex(fi(0.5, 1, 16, 15))");
        match text {
            Value::Char(s) => assert_eq!(s, "4000"),
            other => panic!("{other}"),
        }
    }

    #[test]
    fn hil_loopback_script() {
        let session = exec(
            "h = hil.open('loopback'); hil.regwrite(h, 0, 42); v = hil.regread(h, 0); hil.write(h, [1 2 3]); y = hil.read(h, 3);",
        )
        .unwrap();
        assert!((session.get("v").unwrap().real_scalar().unwrap() - 42.0).abs() < 1e-9);
        let y = session.get("y").unwrap().clone().into_matrix().unwrap().reals().unwrap();
        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn plot_records_a_series() {
        let session = exec("t = linspace(0, 1, 4); plot(t, sin(t)); title('sine');").unwrap();
        assert_eq!(session.figures[0].axes[0].series.len(), 1);
        assert_eq!(session.figures[0].axes[0].title, "sine");
    }
}
