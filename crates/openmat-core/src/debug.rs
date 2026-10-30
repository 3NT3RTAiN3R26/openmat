use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Condvar, Mutex};

use crate::error::{MatError, MatResult};
use crate::plot::Figure;
use crate::session::Session;
use crate::value::Value;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DebugMode {
    Run,
    Step,
}

struct DebugInner {
    breakpoints: HashSet<usize>,
    mode: DebugMode,
    stop: bool,
    resume: bool,
    waiting: bool,
    /// The line we just continued from, so a breakpoint there does not fire again immediately.
    skip_line: Option<usize>,
}

#[derive(Clone)]
pub struct DebugView {
    pub paused_line: Option<usize>,
    pub vars: HashMap<String, Value>,
    pub output: String,
    pub figures: Vec<Figure>,
    pub current_figure: usize,
}

impl Default for DebugView {
    fn default() -> Self {
        Self {
            paused_line: None,
            vars: HashMap::new(),
            output: String::new(),
            figures: Vec::new(),
            current_figure: 0,
        }
    }
}

pub struct DebugCtl {
    inner: Mutex<DebugInner>,
    cvar: Condvar,
    view: Mutex<DebugView>,
}

impl DebugCtl {
    pub fn new(breakpoints: HashSet<usize>, step: bool) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(DebugInner {
                breakpoints,
                mode: if step { DebugMode::Step } else { DebugMode::Run },
                stop: false,
                resume: false,
                waiting: false,
                skip_line: None,
            }),
            cvar: Condvar::new(),
            view: Mutex::new(DebugView::default()),
        })
    }

    pub fn set_breakpoints(&self, breakpoints: HashSet<usize>) {
        self.inner.lock().expect("debug lock").breakpoints = breakpoints;
    }

    pub fn request_step(&self) {
        let mut guard = self.inner.lock().expect("debug lock");
        guard.mode = DebugMode::Step;
        guard.stop = false;
        if guard.waiting {
            guard.resume = true;
            self.cvar.notify_one();
        }
    }

    pub fn request_continue(&self) {
        let mut guard = self.inner.lock().expect("debug lock");
        guard.mode = DebugMode::Run;
        guard.stop = false;
        if guard.waiting {
            guard.resume = true;
            self.cvar.notify_one();
        }
    }

    pub fn request_stop(&self) {
        let mut guard = self.inner.lock().expect("debug lock");
        guard.stop = true;
        guard.resume = true;
        self.cvar.notify_one();
    }

    pub fn view(&self) -> DebugView {
        self.view.lock().expect("debug view").clone()
    }

    pub fn wait(&self, session: &Session, line: usize) -> MatResult<()> {
        let mut guard = self.inner.lock().expect("debug lock");
        if guard.stop {
            return Err(MatError::new("stopped"));
        }
        let breakpoint = guard.breakpoints.contains(&line) && guard.skip_line != Some(line);
        let pause = guard.mode == DebugMode::Step || breakpoint;
        if !pause {
            if guard.skip_line.is_some() && guard.skip_line != Some(line) {
                guard.skip_line = None;
            }
            return Ok(());
        }

        {
            let mut view = self.view.lock().expect("debug view");
            view.paused_line = Some(line);
            view.vars = session.vars.clone();
            view.output = session.output.clone();
            view.figures = session.figures.clone();
            view.current_figure = session.current_figure;
        }

        guard.waiting = true;
        while !guard.resume && !guard.stop {
            guard = self.cvar.wait(guard).expect("debug wait");
        }
        let stop = guard.stop;
        guard.resume = false;
        guard.waiting = false;
        guard.skip_line = Some(line);
        drop(guard);
        if stop {
            return Err(MatError::new("stopped"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    fn wait_until_paused(ctl: &DebugCtl, line: usize) {
        for _ in 0..100 {
            if ctl.view().paused_line == Some(line) {
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        panic!("did not pause at line {line}");
    }

    #[test]
    fn breakpoint_pauses_before_that_line() {
        let ctl = DebugCtl::new(HashSet::from([2]), false);
        let bg = Arc::clone(&ctl);
        let watcher = thread::spawn(move || {
            wait_until_paused(&bg, 2);
            let view = bg.view();
            assert!(view.vars.contains_key("a"));
            assert!(!view.vars.contains_key("b"));
            bg.request_continue();
        });

        let mut session = Session::new();
        session.debug = Some(ctl);
        session.eval_source("a = 1\nb = 2\n").unwrap();
        watcher.join().unwrap();
        assert!(session.vars.contains_key("b"));
    }

    #[test]
    fn step_stops_on_every_statement() {
        let ctl = DebugCtl::new(HashSet::new(), true);
        let bg = Arc::clone(&ctl);
        let watcher = thread::spawn(move || {
            wait_until_paused(&bg, 1);
            assert!(!bg.view().vars.contains_key("a"));
            bg.request_step();
            wait_until_paused(&bg, 2);
            assert!(bg.view().vars.contains_key("a"));
            bg.request_continue();
        });

        let mut session = Session::new();
        session.debug = Some(ctl);
        session.eval_source("a = 1\nb = 2\n").unwrap();
        watcher.join().unwrap();
    }
}
