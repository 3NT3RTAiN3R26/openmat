#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeriesKind {
    Line,
    Stem,
}

#[derive(Debug, Clone)]
pub struct Series {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub kind: SeriesKind,
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct Axes {
    pub series: Vec<Series>,
    pub title: String,
    pub xlabel: String,
    pub ylabel: String,
    pub hold: bool,
}

impl Axes {
    pub fn new() -> Self {
        Self {
            series: Vec::new(),
            title: String::new(),
            xlabel: String::new(),
            ylabel: String::new(),
            hold: false,
        }
    }
}

impl Default for Axes {
    fn default() -> Self {
        Self::new()
    }
}

/// How a newly created figure is shown. This is MATLAB's `DefaultFigureWindowStyle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FigureWindowStyle {
    /// A separate desktop window. MATLAB's `'normal'`.
    Normal,
    /// A tab in one shared figure window. MATLAB's `'docked'`.
    #[default]
    Docked,
}

#[derive(Debug, Clone)]
pub struct Figure {
    pub axes: Vec<Axes>,
    pub rows: usize,
    pub cols: usize,
    pub active: usize,
    /// Style captured when this figure was created.
    pub window_style: FigureWindowStyle,
}

impl Figure {
    pub fn new() -> Self {
        Self {
            axes: vec![Axes::new()],
            rows: 1,
            cols: 1,
            active: 0,
            window_style: FigureWindowStyle::Docked,
        }
    }

    pub fn axes_mut(&mut self) -> &mut Axes {
        let i = self.active.min(self.axes.len().saturating_sub(1));
        &mut self.axes[i]
    }
}

impl Default for Figure {
    fn default() -> Self {
        Self::new()
    }
}
