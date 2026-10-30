use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use openmat_lib::Session;

const SAMPLE: &str = "t = linspace(0, 2*pi, 64);\ny = sin(t);\nplot(t, y);\ntitle('sine');\nxlabel('t');\n";

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "openmat",
        options,
        Box::new(|_cc| Ok(Box::new(OpenmatApp::default()))),
    )
}

struct OpenmatApp {
    code: String,
    console: String,
    status: String,
    session: Session,
}

impl Default for OpenmatApp {
    fn default() -> Self {
        Self {
            code: SAMPLE.to_string(),
            console: String::new(),
            status: String::new(),
            session: openmat_lib::session(),
        }
    }
}

impl eframe::App for OpenmatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("openmat");
                if ui.button("Run").clicked() {
                    self.run_code();
                }
                if ui.button("Clear console").clicked() {
                    self.console.clear();
                    self.status.clear();
                }
            });
            ui.label("MATLAB ist eine Marke von MathWorks. openmat ist nicht mit MathWorks verbunden.");
        });

        egui::SidePanel::right("workspace")
            .resizable(true)
            .default_width(240.0)
            .show(ctx, |ui| {
                ui.heading("Workspace");
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut names: Vec<_> = self.session.vars.keys().cloned().collect();
                    names.sort();
                    for name in names {
                        if matches!(
                            name.as_str(),
                            "pi" | "i" | "j" | "inf" | "Inf" | "nan" | "NaN" | "true" | "false"
                        ) {
                            continue;
                        }
                        if let Some(value) = self.session.vars.get(&name) {
                            ui.label(format!("{name}: {}", value.summary()));
                        }
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Editor");
            egui::ScrollArea::vertical().max_height(240.0).show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut self.code)
                        .desired_width(f32::INFINITY)
                        .desired_rows(12)
                        .code_editor()
                        .font(egui::TextStyle::Monospace),
                );
            });
            if !self.status.is_empty() {
                ui.colored_label(egui::Color32::from_rgb(180, 40, 40), &self.status);
            }
            ui.separator();
            ui.label("Console");
            egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut self.console)
                        .desired_width(f32::INFINITY)
                        .desired_rows(6)
                        .interactive(false)
                        .font(egui::TextStyle::Monospace),
                );
            });
            ui.separator();
            ui.label("Plot");
            self.draw_plots(ui);
        });
    }
}

impl OpenmatApp {
    fn run_code(&mut self) {
        self.session.output.clear();
        match self.session.eval_source(&self.code) {
            Ok(()) => {
                self.console.push_str(&self.session.output);
                self.status.clear();
            }
            Err(err) => {
                self.console.push_str(&self.session.output);
                self.status = err.to_string();
            }
        }
    }

    fn draw_plots(&self, ui: &mut egui::Ui) {
        let Some(figure) = self.session.figures.get(self.session.current_figure) else {
            ui.label("No figure yet.");
            return;
        };
        let mut any = false;
        for (index, axes) in figure.axes.iter().enumerate() {
            if axes.series.is_empty() {
                continue;
            }
            any = true;
            let title = if axes.title.is_empty() {
                format!("axes {}", index + 1)
            } else {
                axes.title.clone()
            };
            ui.label(format!(
                "{title}    {}    {}",
                axes.xlabel, axes.ylabel
            ));
            Plot::new(format!("plot-{index}")).height(220.0).show(ui, |plot_ui| {
                for series in &axes.series {
                    let points: PlotPoints = series
                        .xs
                        .iter()
                        .zip(series.ys.iter())
                        .map(|(x, y)| [*x, *y])
                        .collect();
                    plot_ui.line(Line::new(points).name(if series.label.is_empty() {
                        "series"
                    } else {
                        series.label.as_str()
                    }));
                }
            });
        }
        if !any {
            ui.label("Run a script with plot, stem, step, or bode.");
        }
    }
}
