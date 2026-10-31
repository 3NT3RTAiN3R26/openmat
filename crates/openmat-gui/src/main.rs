use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};

use eframe::egui;
use openmat_lib::{
    command_is_complete, format_number, format_source, lint_source, DebugCtl, Diagnostic,
    FuncHandle, Matrix, Session, Severity, Value,
};

const SAMPLE: &str = "t = linspace(0, 2*pi, 64);\ny = sin(t);\nplot(t, y);\ntitle('sine');\nxlabel('t');\n";

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1360.0, 800.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "openmat",
        options,
        Box::new(|_cc| Ok(Box::new(OpenmatApp::from_args()))),
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FileSort {
    Name,
    Size,
    Modified,
    Type,
}

impl FileSort {
    fn from_index(index: usize) -> Self {
        match index {
            1 => Self::Size,
            2 => Self::Modified,
            3 => Self::Type,
            _ => Self::Name,
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Name => 0,
            Self::Size => 1,
            Self::Modified => 2,
            Self::Type => 3,
        }
    }
}

struct FileEntry {
    name: String,
    path: PathBuf,
    is_dir: bool,
    size: u64,
    modified: Option<SystemTime>,
    kind: String,
}

struct DebugDone {
    session: Session,
    result: Result<(), String>,
}

struct DebugRun {
    ctl: std::sync::Arc<DebugCtl>,
    done_rx: Receiver<DebugDone>,
    handle: Option<JoinHandle<()>>,
}

struct OpenmatApp {
    code: String,
    console: String,
    status: String,
    session: Session,
    path: Option<PathBuf>,
    folder: PathBuf,
    file_sort: FileSort,
    file_sort_asc: bool,
    selected_file: Option<PathBuf>,
    selected_var: Option<String>,
    inspected: Option<String>,
    breakpoints: HashSet<usize>,
    debug_line: Option<usize>,
    debug_run: Option<DebugRun>,
    console_base: String,
    new_figure: bool,
    docked_tab: usize,
    command: String,
    command_pending: String,
    command_history: Vec<String>,
    command_history_pos: Option<usize>,
}

impl OpenmatApp {
    fn from_args() -> Self {
        let mut app = Self {
            code: SAMPLE.to_string(),
            console: String::new(),
            status: String::new(),
            session: openmat_lib::session(),
            path: None,
            folder: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            file_sort: FileSort::Name,
            file_sort_asc: true,
            selected_file: None,
            selected_var: None,
            inspected: None,
            breakpoints: HashSet::new(),
            debug_line: None,
            debug_run: None,
            console_base: String::new(),
            new_figure: false,
            docked_tab: 0,
            command: String::new(),
            command_pending: String::new(),
            command_history: Vec::new(),
            command_history_pos: None,
        };
        if let Some(arg) = std::env::args().nth(1) {
            app.load_path(PathBuf::from(arg));
        }
        app
    }
}

impl eframe::App for OpenmatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_debug(ctx);
        let command = |key| ctx.input(|i| i.modifiers.command && i.key_pressed(key));
        let open_shortcut = command(egui::Key::O);
        let save_shortcut = command(egui::Key::S);
        let format_shortcut = command(egui::Key::I);
        let dropped = dropped_script(ctx);
        if open_shortcut {
            self.pick_file();
        } else if save_shortcut {
            self.save_file();
        } else if format_shortcut {
            self.format_editor();
        } else if let Some(path) = dropped {
            self.load_path(path);
        }

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("openmat");
                if ui.button("Open…").clicked() {
                    self.pick_file();
                }
                if ui.button("Save").on_hover_text("Ctrl+S").clicked() {
                    self.save_file();
                }
                if ui.button("Format").on_hover_text("Ctrl+I").clicked() {
                    self.format_editor();
                }
                if ui
                    .add(egui::Button::new("Run").fill(egui::Color32::from_rgb(46, 120, 64)))
                    .clicked()
                {
                    self.debug_run_or_continue(false);
                }
                if ui
                    .add(egui::Button::new("Step").fill(egui::Color32::from_rgb(36, 78, 140)))
                    .clicked()
                {
                    self.debug_run_or_continue(true);
                }
                if ui
                    .add_enabled(
                        self.debug_run.is_some(),
                        egui::Button::new("Stop").fill(egui::Color32::from_rgb(140, 48, 48)),
                    )
                    .clicked()
                {
                    self.debug_stop();
                }
                if ui.button("Clear console").clicked() {
                    self.console.clear();
                    self.command_pending.clear();
                    self.status.clear();
                }
                ui.separator();
                ui.label("Options");
                ui.radio_value(&mut self.new_figure, false, "Reuse Figure");
                ui.radio_value(&mut self.new_figure, true, "New Figure");
            });
            ui.label("MATLAB ist eine Marke von MathWorks. openmat ist nicht mit MathWorks verbunden.");
        });

        egui::SidePanel::left("files")
            .resizable(true)
            .default_width(380.0)
            .min_width(260.0)
            .show(ctx, |ui| {
                ui.heading("Files")
                    .on_hover_text("Double-click a folder to open it, or a .m file to edit it.");
                self.show_files(ui);
            });

        egui::SidePanel::right("workspace")
            .resizable(true)
            .default_width(360.0)
            .min_width(260.0)
            .show(ctx, |ui| {
                ui.heading("Workspace")
                    .on_hover_text("Double-click a variable to inspect its contents.");
                self.show_workspace(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(match &self.path {
                Some(path) => format!("Editor  {}", path.display()),
                None => "Editor".to_string(),
            });
            let current_line = self.debug_line;
            let editable = self.debug_run.is_none();
            let breakpoints = self.breakpoints.clone();
            let mut toggled = None;
            egui::ScrollArea::vertical()
                .id_salt("editor")
                .max_height(240.0)
                .show(ui, |ui| {
                    toggled = show_code_editor(
                        ui,
                        &mut self.code,
                        &breakpoints,
                        current_line,
                        editable,
                    );
                });
            if let Some(line) = toggled {
                if !self.breakpoints.insert(line) {
                    self.breakpoints.remove(&line);
                }
                if let Some(run) = &self.debug_run {
                    run.ctl.set_breakpoints(self.breakpoints.clone());
                }
            }
            self.show_lint(ui);
            if !self.status.is_empty() {
                ui.colored_label(egui::Color32::from_rgb(180, 40, 40), &self.status);
            }
            ui.separator();
            self.show_command_window(ui);
        });

        self.show_figure_windows(ctx);
        self.show_variable_window(ctx);
    }
}

fn dropped_script(ctx: &egui::Context) -> Option<PathBuf> {
    ctx.input(|input| {
        let paths: Vec<PathBuf> = input
            .raw
            .dropped_files
            .iter()
            .filter_map(|file| file.path.clone())
            .collect();
        paths
            .iter()
            .find(|path| is_m_file(path))
            .cloned()
            .or_else(|| paths.into_iter().next())
    })
}

fn show_code_editor(
    ui: &mut egui::Ui,
    code: &mut String,
    breakpoints: &HashSet<usize>,
    debug_line: Option<usize>,
    editable: bool,
) -> Option<usize> {
    let mut toggled = None;
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        let font_id = egui::TextStyle::Monospace.resolve(ui.style());
        let line_count = code.split('\n').count().max(1);
        let digits = line_count.to_string().len().max(2);
        let char_width = ui.fonts(|fonts| fonts.glyph_width(&font_id, '0'));
        let gutter_width = char_width * (digits as f32 + 2.4);
        let (_, gutter_rect) = ui.allocate_space(egui::vec2(gutter_width, 0.0));

        let output = egui::TextEdit::multiline(code)
            .id_salt("editor-text")
            .desired_width(ui.available_width())
            .desired_rows(12)
            .code_editor()
            .interactive(editable)
            .font(egui::TextStyle::Monospace)
            .show(ui);

        let frame = output.response.rect.expand2(egui::vec2(4.0, 2.0));
        let gutter = egui::Rect::from_min_max(
            egui::pos2(gutter_rect.left(), frame.top()),
            egui::pos2(gutter_rect.right(), frame.bottom()),
        );
        let painter = ui.painter_at(gutter);
        painter.rect_filled(
            gutter,
            egui::Rounding::ZERO,
            ui.visuals().extreme_bg_color,
        );
        painter.vline(
            gutter.right(),
            gutter.y_range(),
            ui.visuals().widgets.noninteractive.bg_stroke,
        );

        let dim = ui.visuals().weak_text_color();
        let strong = ui.visuals().widgets.active.text_color();
        let current_line = output
            .response
            .has_focus()
            .then_some(output.cursor_range)
            .flatten()
            .map(|range| line_at_char(code, range.primary.ccursor.index));

        let gutter_click = ui.interact(gutter, ui.id().with("gutter-click"), egui::Sense::click());
        gutter_click.clone().on_hover_text("Click a line number to toggle a breakpoint");
        if gutter_click.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }

        let mut line_no = 1usize;
        let mut show_number = true;
        for row in &output.galley.rows {
            if show_number {
                let y = output.galley_pos.y + row.rect.min.y;
                let row_height = row.rect.height().max(1.0);
                if debug_line == Some(line_no) {
                    let bar = egui::Rect::from_min_size(
                        egui::pos2(frame.left(), y),
                        egui::vec2(frame.width(), row_height),
                    );
                    ui.painter_at(frame).rect_filled(
                        bar,
                        egui::Rounding::ZERO,
                        egui::Color32::from_rgba_unmultiplied(70, 130, 210, 70),
                    );
                }
                if breakpoints.contains(&line_no) {
                    painter.circle_filled(
                        egui::pos2(gutter.left() + 7.0, y + row_height * 0.5),
                        4.0,
                        egui::Color32::from_rgb(210, 48, 48),
                    );
                }
                painter.text(
                    egui::pos2(gutter.right() - char_width * 0.6, y),
                    egui::Align2::RIGHT_TOP,
                    line_no.to_string(),
                    font_id.clone(),
                    if debug_line == Some(line_no) || current_line == Some(line_no) {
                        strong
                    } else {
                        dim
                    },
                );
                if gutter_click.clicked() {
                    if let Some(pointer) = gutter_click.interact_pointer_pos() {
                        if pointer.y >= y && pointer.y < y + row_height {
                            toggled = Some(line_no);
                        }
                    }
                }
                line_no += 1;
            }
            show_number = row.ends_with_newline;
        }
    });
    toggled
}

fn line_at_char(text: &str, char_index: usize) -> usize {
    text.chars().take(char_index).filter(|c| *c == '\n').count() + 1
}

fn lint_color(diagnostic: &Diagnostic) -> egui::Color32 {
    match diagnostic.severity {
        Severity::Error => egui::Color32::from_rgb(180, 40, 40),
        Severity::Warning => egui::Color32::from_rgb(180, 120, 30),
    }
}

fn join_console(previous: &str, output: &str, homed: bool) -> String {
    if homed {
        output.to_string()
    } else {
        format!("{previous}{output}")
    }
}

fn lint_line(diagnostic: &Diagnostic) -> String {
    format!("line {}: {}", diagnostic.line, diagnostic.message)
}

fn is_m_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("m"))
}

fn is_constant(name: &str) -> bool {
    matches!(
        name,
        "pi" | "i" | "j" | "inf" | "Inf" | "nan" | "NaN" | "true" | "false"
    )
}

fn file_widths(total: f32) -> [f32; 4] {
    let size = 72.0;
    let date = 132.0;
    let kind = 72.0;
    let name = (total - size - date - kind).max(80.0);
    [name, size, date, kind]
}

fn workspace_widths(total: f32) -> [f32; 4] {
    let value = 120.0;
    let size = 72.0;
    let class = 110.0;
    let name = (total - value - size - class).max(72.0);
    [name, value, size, class]
}

fn table_header(
    ui: &mut egui::Ui,
    id: &str,
    widths: &[f32],
    labels: &[&str],
    sort_col: Option<usize>,
    ascending: bool,
) -> Option<usize> {
    let row_w = widths.iter().sum::<f32>();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(row_w, 22.0), egui::Sense::hover());
    let font = egui::TextStyle::Small.resolve(ui.style());
    let mut clicked = None;
    let mut x = rect.left();
    for (index, label) in labels.iter().enumerate() {
        let width = widths[index];
        let cell = egui::Rect::from_min_size(egui::pos2(x, rect.top()), egui::vec2(width, rect.height()));
        let response = ui.interact(cell, ui.id().with((id, index)), egui::Sense::click());
        let text = if sort_col == Some(index) {
            format!("{label} {}", if ascending { "^" } else { "v" })
        } else {
            (*label).to_string()
        };
        let color = if response.hovered() {
            ui.visuals().strong_text_color()
        } else {
            ui.visuals().weak_text_color()
        };
        ui.painter_at(cell).text(
            cell.left_center() + egui::vec2(4.0, 0.0),
            egui::Align2::LEFT_CENTER,
            text,
            font.clone(),
            color,
        );
        if response.clicked() {
            clicked = Some(index);
        }
        x += width;
    }
    ui.painter().hline(
        rect.x_range(),
        rect.bottom(),
        ui.visuals().widgets.noninteractive.bg_stroke,
    );
    clicked
}

fn table_row(ui: &mut egui::Ui, widths: &[f32], cells: &[String], selected: bool) -> egui::Response {
    let row_w = widths.iter().sum::<f32>();
    let (rect, response) = ui.allocate_exact_size(egui::vec2(row_w, 20.0), egui::Sense::click());
    if selected {
        ui.painter()
            .rect_filled(rect, egui::Rounding::ZERO, ui.visuals().selection.bg_fill);
    } else if response.hovered() {
        ui.painter().rect_filled(
            rect,
            egui::Rounding::ZERO,
            ui.visuals().widgets.hovered.bg_fill,
        );
    }
    let color = if selected {
        ui.visuals().selection.stroke.color
    } else {
        ui.visuals().text_color()
    };
    let font = egui::TextStyle::Body.resolve(ui.style());
    let mut x = rect.left();
    for (index, cell) in cells.iter().enumerate() {
        let width = widths.get(index).copied().unwrap_or(40.0);
        let cell_rect =
            egui::Rect::from_min_size(egui::pos2(x, rect.top()), egui::vec2(width, rect.height()));
        ui.painter_at(cell_rect).text(
            cell_rect.left_center() + egui::vec2(4.0, 0.0),
            egui::Align2::LEFT_CENTER,
            cell,
            font.clone(),
            color,
        );
        x += width;
    }
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

fn read_folder(folder: &Path) -> Result<Vec<FileEntry>, String> {
    let mut entries = Vec::new();
    let dir = std::fs::read_dir(folder)
        .map_err(|err| format!("cannot read {}: {err}", folder.display()))?;
    for item in dir {
        let item = match item {
            Ok(item) => item,
            Err(_) => continue,
        };
        let name = item.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let meta = item.metadata().ok();
        let is_dir = meta.as_ref().is_some_and(|meta| meta.is_dir());
        let kind = if is_dir {
            "Folder".to_string()
        } else {
            item.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.to_ascii_lowercase())
                .filter(|ext| !ext.is_empty())
                .unwrap_or_else(|| "File".to_string())
        };
        entries.push(FileEntry {
            name,
            path: item.path(),
            is_dir,
            size: meta.as_ref().map(|meta| meta.len()).unwrap_or(0),
            modified: meta.as_ref().and_then(|meta| meta.modified().ok()),
            kind,
        });
    }
    Ok(entries)
}

fn sort_entries(entries: &mut [FileEntry], sort: FileSort, ascending: bool) {
    entries.sort_by(|a, b| {
        let dirs = b.is_dir.cmp(&a.is_dir);
        if dirs != std::cmp::Ordering::Equal {
            return dirs;
        }
        let order = match sort {
            FileSort::Name => cmp_name(&a.name, &b.name),
            FileSort::Size => a.size.cmp(&b.size).then_with(|| cmp_name(&a.name, &b.name)),
            FileSort::Modified => a
                .modified
                .cmp(&b.modified)
                .then_with(|| cmp_name(&a.name, &b.name)),
            FileSort::Type => a.kind.cmp(&b.kind).then_with(|| cmp_name(&a.name, &b.name)),
        };
        if ascending {
            order
        } else {
            order.reverse()
        }
    });
}

fn cmp_name(a: &str, b: &str) -> std::cmp::Ordering {
    a.to_lowercase().cmp(&b.to_lowercase())
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{} KB", bytes.div_ceil(1024))
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn format_modified(time: SystemTime) -> String {
    let utc: chrono::DateTime<chrono::Utc> = time.into();
    utc.with_timezone(&chrono::Local)
        .format("%d.%m.%Y %H:%M")
        .to_string()
}

fn value_column(value: &Value) -> String {
    match value {
        Value::Matrix(matrix) => matrix_preview(matrix),
        Value::Char(text) if text.chars().count() <= 24 => format!("'{text}'"),
        Value::Func(FuncHandle::Named(name)) => format!("@{name}"),
        Value::Func(FuncHandle::Anon { params, .. }) => format!("@({})", params.join(", ")),
        _ => String::new(),
    }
}

/// MATLAB workspace preview: elements separated by `,`, rows by `;`.
fn matrix_preview(matrix: &Matrix) -> String {
    if matrix.numel() == 0 {
        return String::new();
    }
    if matrix.numel() == 1 {
        return cell_text(matrix, 0, 0);
    }
    if matrix.rows > 6 || matrix.cols > 6 {
        return String::new();
    }
    let rows: Vec<String> = (0..matrix.rows)
        .map(|row| {
            (0..matrix.cols)
                .map(|col| cell_text(matrix, row, col))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();
    format!("[{}]", rows.join(";"))
}

fn size_text(value: &Value) -> String {
    match value {
        Value::Matrix(matrix) => format!("{}x{}", matrix.rows, matrix.cols),
        Value::Char(text) => format!("1x{}", text.chars().count()),
        Value::Cell { rows, cols, .. } => format!("{rows}x{cols}"),
        Value::Fi(fi) => format!("{}x{}", fi.rows, fi.cols),
        Value::Struct(_) | Value::System(_) | Value::Func(_) => "1x1".into(),
        Value::Multi(items) => format!("1x{}", items.len()),
    }
}

fn class_name(value: &Value) -> String {
    match value {
        Value::Matrix(matrix) if matrix.logical => "logical".into(),
        Value::Matrix(matrix) if !matrix.is_real() => "complex double".into(),
        Value::Matrix(_) => "double".into(),
        other => other.type_name().into(),
    }
}

fn show_value(ui: &mut egui::Ui, name: &str, value: &Value) {
    match value {
        Value::Matrix(matrix) => show_matrix(ui, name, matrix),
        Value::Fi(fi) => {
            ui.label(format!(
                "fi {}{},{}",
                if fi.signed { "s" } else { "u" },
                fi.word,
                fi.frac
            ));
            show_matrix(ui, name, &fi.to_matrix());
        }
        Value::Char(text) => {
            let mut text = text.clone();
            ui.add(
                egui::TextEdit::multiline(&mut text)
                    .id_salt(("var-char", name))
                    .interactive(false)
                    .desired_width(f32::INFINITY)
                    .desired_rows(8)
                    .font(egui::TextStyle::Monospace),
            );
        }
        Value::Cell { rows, cols, data } => show_cell(ui, name, *rows, *cols, data),
        Value::Struct(fields) => {
            egui::ScrollArea::vertical()
                .id_salt(("var-struct", name))
                .max_height(420.0)
                .show(ui, |ui| {
                    egui::Grid::new(("var-struct-grid", name))
                        .striped(true)
                        .show(ui, |ui| {
                            ui.weak("Field");
                            ui.weak("Value");
                            ui.end_row();
                            for (field, field_value) in fields {
                                ui.monospace(field);
                                ui.label(field_value.summary());
                                ui.end_row();
                            }
                        });
                });
        }
        other => {
            egui::ScrollArea::vertical()
                .id_salt(("var-text", name))
                .max_height(420.0)
                .show(ui, |ui| {
                    ui.monospace(other.to_string());
                });
        }
    }
}

fn show_matrix(ui: &mut egui::Ui, name: &str, matrix: &Matrix) {
    if matrix.numel() == 0 {
        ui.weak("Empty matrix.");
        return;
    }
    let rows = matrix.rows.min(200);
    let cols = matrix.cols.min(40);
    if rows < matrix.rows || cols < matrix.cols {
        ui.label(format!(
            "Showing {rows}×{cols} of {}×{}.",
            matrix.rows, matrix.cols
        ));
    }
    egui::ScrollArea::both()
        .id_salt(("var-matrix", name))
        .auto_shrink([false, false])
        .max_height(420.0)
        .min_scrolled_height(180.0)
        .show(ui, |ui| {
            egui::Grid::new(("var-matrix-grid", name))
                .striped(true)
                .spacing([12.0, 3.0])
                .show(ui, |ui| {
                    ui.weak("");
                    for col in 0..cols {
                        ui.weak(format!("{}", col + 1));
                    }
                    ui.end_row();
                    for row in 0..rows {
                        ui.weak(format!("{}", row + 1));
                        for col in 0..cols {
                            ui.monospace(cell_text(matrix, row, col));
                        }
                        ui.end_row();
                    }
                });
        });
}

fn show_cell(ui: &mut egui::Ui, name: &str, rows: usize, cols: usize, data: &[Value]) {
    let shown_rows = rows.min(40);
    let shown_cols = cols.min(12);
    egui::ScrollArea::both()
        .id_salt(("var-cell", name))
        .max_height(420.0)
        .show(ui, |ui| {
            egui::Grid::new(("var-cell-grid", name))
                .striped(true)
                .spacing([12.0, 3.0])
                .show(ui, |ui| {
                    ui.weak("");
                    for col in 0..shown_cols {
                        ui.weak(format!("{}", col + 1));
                    }
                    ui.end_row();
                    for row in 0..shown_rows {
                        ui.weak(format!("{}", row + 1));
                        for col in 0..shown_cols {
                            let index = col * rows + row;
                            ui.label(data.get(index).map(Value::summary).unwrap_or_default());
                        }
                        ui.end_row();
                    }
                });
        });
}

fn cell_text(matrix: &Matrix, row: usize, col: usize) -> String {
    let value = matrix.get(row, col);
    if matrix.logical {
        if value.re != 0.0 {
            "1".to_string()
        } else {
            "0".to_string()
        }
    } else if value.im.abs() < 1e-12 {
        format_number(value.re)
    } else if value.im >= 0.0 {
        format!("{}+{}i", format_number(value.re), format_number(value.im))
    } else {
        format!("{}{}i", format_number(value.re), format_number(value.im))
    }
}

impl OpenmatApp {
    fn pick_file(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .add_filter("MATLAB script", &["m"])
            .add_filter("All files", &["*"]);
        if let Some(dir) = self.path.as_ref().and_then(|path| path.parent()) {
            dialog = dialog.set_directory(dir);
        }
        if let Some(path) = dialog.pick_file() {
            self.load_path(path);
        }
    }

    fn show_command_window(&mut self, ui: &mut egui::Ui) {
        ui.label("Command Window");
        egui::ScrollArea::vertical()
            .id_salt("console")
            .stick_to_bottom(true)
            .max_height(140.0)
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut self.console)
                        .id_salt("console-log")
                        .desired_width(f32::INFINITY)
                        .desired_rows(6)
                        .interactive(false)
                        .font(egui::TextStyle::Monospace),
                );
            });
        if self.debug_run.is_some() {
            ui.weak("Stop the debugger to use the command window.");
            return;
        }
        let prompt = if self.command_pending.is_empty() { ">>" } else { ".." };
        let mut submit = false;
        let mut older = false;
        let mut newer = false;
        ui.horizontal(|ui| {
            ui.monospace(prompt);
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.command)
                    .id_salt("command-line")
                    .desired_width(f32::INFINITY)
                    .font(egui::TextStyle::Monospace),
            );
            let enter = ui.input(|input| {
                input.key_pressed(egui::Key::Enter) && !input.modifiers.shift
            });
            older = response.has_focus() && ui.input(|input| input.key_pressed(egui::Key::ArrowUp));
            newer = response.has_focus() && ui.input(|input| input.key_pressed(egui::Key::ArrowDown));
            submit = response.has_focus() && enter;
        });
        if older {
            self.history_older();
        } else if newer {
            self.history_newer();
        } else if submit {
            self.submit_command();
        }
    }

    fn submit_command(&mut self) {
        let line = std::mem::take(&mut self.command);
        let prompt = if self.command_pending.is_empty() { ">>" } else { ".." };
        if !self.console.is_empty() && !self.console.ends_with('\n') {
            self.console.push('\n');
        }
        self.console.push_str(&format!("{prompt} {line}\n"));
        let full = if self.command_pending.is_empty() {
            line
        } else {
            format!("{}\n{line}", self.command_pending)
        };
        if !command_is_complete(&full) {
            self.command_pending = full;
            return;
        }
        self.command_pending.clear();
        if full.trim().is_empty() {
            return;
        }
        self.command_history.push(full.clone());
        self.command_history_pos = None;
        self.session.new_figure_on_plot = self.new_figure;
        self.session.output.clear();
        match self.session.eval_source(&full) {
            Ok(()) => self.append_command_output(),
            Err(err) => {
                self.append_command_output();
                if !self.console.ends_with('\n') {
                    self.console.push('\n');
                }
                self.console.push_str(&err.to_string());
                self.console.push('\n');
            }
        }
    }

    fn append_command_output(&mut self) {
        if self.session.output_homed {
            self.console = self.session.output.clone();
            return;
        }
        if self.session.output.is_empty() {
            return;
        }
        self.console.push_str(&self.session.output);
        if !self.session.output.ends_with('\n') {
            self.console.push('\n');
        }
    }

    fn history_older(&mut self) {
        if self.command_history.is_empty() || !self.command_pending.is_empty() {
            return;
        }
        let next = match self.command_history_pos {
            None => self.command_history.len() - 1,
            Some(0) => 0,
            Some(index) => index - 1,
        };
        self.command_history_pos = Some(next);
        self.command = self.command_history[next].clone();
    }

    fn history_newer(&mut self) {
        let Some(index) = self.command_history_pos else {
            return;
        };
        if index + 1 >= self.command_history.len() {
            self.command_history_pos = None;
            self.command.clear();
        } else {
            let next = index + 1;
            self.command_history_pos = Some(next);
            self.command = self.command_history[next].clone();
        }
    }

    fn save_file(&mut self) {
        if self.debug_run.is_some() {
            self.status = "Stop the debugger before saving.".into();
            return;
        }
        let Some(path) = self.path.clone().or_else(|| self.pick_save_path()) else {
            return;
        };
        match std::fs::write(&path, &self.code) {
            Ok(()) => {
                if let Some(parent) = path.parent() {
                    if parent.is_dir() {
                        self.folder = parent.to_path_buf();
                    }
                }
                self.selected_file = Some(path.clone());
                self.path = Some(path);
                self.status.clear();
            }
            Err(err) => {
                self.status = format!("cannot write {}: {err}", path.display());
            }
        }
    }

    fn pick_save_path(&self) -> Option<PathBuf> {
        rfd::FileDialog::new()
            .add_filter("MATLAB script", &["m"])
            .add_filter("All files", &["*"])
            .set_directory(&self.folder)
            .set_file_name("script.m")
            .save_file()
    }

    fn format_editor(&mut self) {
        if self.debug_run.is_some() {
            self.status = "Stop the debugger before formatting.".into();
            return;
        }
        self.code = format_source(&self.code);
        self.status.clear();
    }

    fn show_lint(&self, ui: &mut egui::Ui) {
        let diagnostics = lint_source(&self.code);
        ui.label("Lint");
        if diagnostics.is_empty() {
            ui.weak("No issues.");
            return;
        }
        egui::ScrollArea::vertical()
            .id_salt("lint")
            .max_height(72.0)
            .show(ui, |ui| {
                for diagnostic in diagnostics.iter().take(12) {
                    ui.colored_label(lint_color(diagnostic), lint_line(diagnostic));
                }
            });
    }

    fn load_path(&mut self, path: PathBuf) {
        if self.debug_run.is_some() {
            self.status = "Stop the debugger before opening another file.".into();
            return;
        }
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                if let Some(parent) = path.parent() {
                    if parent.is_dir() {
                        self.folder = parent.to_path_buf();
                    }
                }
                self.selected_file = Some(path.clone());
                self.code = text;
                self.path = Some(path);
                self.status.clear();
            }
            Err(err) => {
                self.status = format!("cannot read {}: {err}", path.display());
            }
        }
    }

    fn show_files(&mut self, ui: &mut egui::Ui) {
        let can_up = self.folder.parent().is_some();
        let mut go_up = false;
        ui.horizontal(|ui| {
            if ui.add_enabled(can_up, egui::Button::new("Up")).clicked() {
                go_up = true;
            }
            ui.label(egui::RichText::new(self.folder.display().to_string()).small());
        });
        if go_up {
            if let Some(parent) = self.folder.parent() {
                self.folder = parent.to_path_buf();
                self.selected_file = None;
            }
        }

        let mut entries = match read_folder(&self.folder) {
            Ok(entries) => entries,
            Err(err) => {
                ui.colored_label(egui::Color32::from_rgb(180, 40, 40), err);
                return;
            }
        };
        sort_entries(&mut entries, self.file_sort, self.file_sort_asc);
        if let Some(parent) = self.folder.parent() {
            entries.insert(
                0,
                FileEntry {
                    name: "..".into(),
                    path: parent.to_path_buf(),
                    is_dir: true,
                    size: 0,
                    modified: None,
                    kind: "Folder".into(),
                },
            );
        }

        let widths = file_widths(ui.available_width());
        if let Some(col) = table_header(
            ui,
            "files-header",
            &widths,
            &["Name", "Size", "Date Modified", "Type"],
            Some(self.file_sort.index()),
            self.file_sort_asc,
        ) {
            let sort = FileSort::from_index(col);
            if self.file_sort == sort {
                self.file_sort_asc = !self.file_sort_asc;
            } else {
                self.file_sort = sort;
                self.file_sort_asc = !matches!(sort, FileSort::Size | FileSort::Modified);
            }
        }

        let mut enter: Option<PathBuf> = None;
        let mut open_path: Option<PathBuf> = None;
        egui::ScrollArea::vertical()
            .id_salt("files")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if entries.is_empty() {
                    ui.weak("This folder is empty.");
                    return;
                }
                for entry in &entries {
                    let cells = [
                        entry.name.clone(),
                        if entry.is_dir || entry.name == ".." {
                            String::new()
                        } else {
                            format_size(entry.size)
                        },
                        entry.modified.map(format_modified).unwrap_or_default(),
                        if entry.name == ".." {
                            String::new()
                        } else {
                            entry.kind.clone()
                        },
                    ];
                    let resp = table_row(
                        ui,
                        &widths,
                        &cells,
                        self.selected_file.as_ref() == Some(&entry.path),
                    );
                    if resp.double_clicked() {
                        if entry.is_dir {
                            enter = Some(entry.path.clone());
                        } else {
                            open_path = Some(entry.path.clone());
                        }
                    } else if resp.clicked() {
                        self.selected_file = Some(entry.path.clone());
                    }
                }
            });

        if let Some(path) = enter {
            self.folder = path;
            self.selected_file = None;
        } else if let Some(path) = open_path {
            self.load_path(path);
        }
    }

    fn show_workspace(&mut self, ui: &mut egui::Ui) {
        let mut names: Vec<_> = self.session.vars.keys().cloned().collect();
        names.sort();
        names.retain(|name| !is_constant(name));
        let rows: Vec<(String, String, String, String)> = names
            .iter()
            .filter_map(|name| {
                let value = self.session.vars.get(name)?;
                Some((
                    name.clone(),
                    value_column(value),
                    size_text(value),
                    class_name(value),
                ))
            })
            .collect();

        let widths = workspace_widths(ui.available_width());
        let _ = table_header(
            ui,
            "workspace-header",
            &widths,
            &["Name", "Value", "Size", "Class"],
            None,
            true,
        );

        let mut inspect: Option<String> = None;
        egui::ScrollArea::vertical()
            .id_salt("workspace-vars")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if rows.is_empty() {
                    ui.weak("Run a script to create variables.");
                    return;
                }
                for (name, value, size, class) in &rows {
                    let resp = table_row(
                        ui,
                        &widths,
                        &[name.clone(), value.clone(), size.clone(), class.clone()],
                        self.selected_var.as_ref() == Some(name),
                    );
                    if resp.double_clicked() {
                        self.selected_var = Some(name.clone());
                        inspect = Some(name.clone());
                    } else if resp.clicked() {
                        self.selected_var = Some(name.clone());
                    }
                }
            });
        if let Some(name) = inspect {
            self.inspected = Some(name);
        }
    }

    fn show_variable_window(&mut self, ctx: &egui::Context) {
        let Some(name) = self.inspected.clone() else {
            return;
        };
        let title = self
            .session
            .vars
            .get(&name)
            .map(|value| format!("{name}    {}", value.summary()))
            .unwrap_or_else(|| name.clone());
        let mut open = true;
        egui::Window::new(title)
            .id(egui::Id::new("variable-viewer"))
            .open(&mut open)
            .resizable(true)
            .default_size([520.0, 360.0])
            .show(ctx, |ui| match self.session.vars.get(&name) {
                Some(value) => show_value(ui, &name, value),
                None => {
                    ui.label(format!("{name} is no longer in the workspace."));
                }
            });
        if !open {
            self.inspected = None;
        }
    }

    fn debug_run_or_continue(&mut self, step: bool) {
        if let Some(run) = &self.debug_run {
            if step {
                run.ctl.request_step();
            } else {
                run.ctl.request_continue();
            }
            return;
        }
        let ctl = DebugCtl::new(self.breakpoints.clone(), step);
        let mut session = std::mem::replace(&mut self.session, openmat_lib::session());
        session.output.clear();
        session.new_figure_on_plot = self.new_figure;
        session.debug = Some(std::sync::Arc::clone(&ctl));
        let code = self.code.clone();
        let (tx, done_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let result = session.eval_source(&code).map_err(|err| err.to_string());
            session.debug = None;
            let _ = tx.send(DebugDone { session, result });
        });
        self.console_base = self.console.clone();
        self.status.clear();
        self.debug_line = None;
        self.debug_run = Some(DebugRun {
            ctl,
            done_rx,
            handle: Some(handle),
        });
    }

    fn debug_stop(&mut self) {
        if let Some(run) = &self.debug_run {
            run.ctl.request_stop();
        }
    }

    fn poll_debug(&mut self, ctx: &egui::Context) {
        let finished = self
            .debug_run
            .as_ref()
            .and_then(|run| run.done_rx.try_recv().ok());
        if let Some(done) = finished {
            if let Some(run) = self.debug_run.take() {
                if let Some(handle) = run.handle {
                    let _ = handle.join();
                }
            }
            self.console = join_console(&self.console_base, &done.session.output, done.session.output_homed);
            self.session = done.session;
            self.debug_line = None;
            match done.result {
                Ok(()) => self.status.clear(),
                Err(err) if err == "stopped" => self.status.clear(),
                Err(err) => self.status = err,
            }
            return;
        }

        let Some(run) = self.debug_run.as_ref() else {
            return;
        };
        let view = run.ctl.view();
        if let Some(line) = view.paused_line {
            self.debug_line = Some(line);
            self.session.vars = view.vars;
            self.session.output = view.output.clone();
            self.session.figures = view.figures;
            self.session.current_figure = view.current_figure;
            self.console = join_console(&self.console_base, &view.output, view.output_homed);
        }
        ctx.request_repaint_after(Duration::from_millis(30));
    }

    fn show_figure_windows(&mut self, ctx: &egui::Context) {
        let open_figures: Vec<(usize, openmat_lib::Figure)> = self
            .session
            .figures
            .iter()
            .enumerate()
            .filter(|(_, figure)| figure.axes.iter().any(|axes| !axes.series.is_empty()))
            .map(|(index, figure)| (index, figure.clone()))
            .collect();
        let mut closed = Vec::new();
        let (docked, normal): (Vec<_>, Vec<_>) = open_figures
            .into_iter()
            .partition(|(_, figure)| figure.window_style == openmat_lib::FigureWindowStyle::Docked);
        for (fig_index, figure) in normal {
            if show_native_figure(ctx, fig_index, &figure) {
                closed.push(fig_index);
            }
        }
        if !docked.is_empty() {
            closed.extend(show_docked_figures(ctx, &docked, &mut self.docked_tab));
        }
        for index in closed {
            if let Some(figure) = self.session.figures.get_mut(index) {
                let style = figure.window_style;
                *figure = openmat_lib::Figure::new();
                figure.window_style = style;
            }
        }
    }
}

fn show_native_figure(
    ctx: &egui::Context,
    fig_index: usize,
    figure: &openmat_lib::Figure,
) -> bool {
    let mut close_requested = false;
    let title = format!("Figure {}", fig_index + 1);
    ctx.show_viewport_immediate(
        egui::ViewportId::from_hash_of(("openmat-figure", fig_index)),
        egui::ViewportBuilder::default()
            .with_title(&title)
            .with_inner_size([640.0, 480.0])
            .with_close_button(true),
        |ctx, class| {
            if ctx.input(|input| input.viewport().close_requested()) {
                close_requested = true;
            }
            let mut embedded_open = true;
            let draw = |ui: &mut egui::Ui| draw_figure(ui, fig_index, figure);
            if class == egui::ViewportClass::Embedded {
                egui::Window::new(&title)
                    .id(egui::Id::new(("figure-window", fig_index)))
                    .open(&mut embedded_open)
                    .resizable(true)
                    .default_size([480.0, 320.0])
                    .default_pos(egui::pos2(
                        48.0 + fig_index as f32 * 32.0,
                        72.0 + fig_index as f32 * 32.0,
                    ))
                    .show(ctx, draw);
                if !embedded_open {
                    close_requested = true;
                }
            } else {
                egui::CentralPanel::default().show(ctx, draw);
            }
        },
    );
    close_requested
}

fn show_docked_figures(
    ctx: &egui::Context,
    figures: &[(usize, openmat_lib::Figure)],
    selected: &mut usize,
) -> Vec<usize> {
    if !figures.iter().any(|(index, _)| *index == *selected) {
        *selected = figures[0].0;
    }
    let mut close_requested = false;
    let mut close_tab = None;
    ctx.show_viewport_immediate(
        egui::ViewportId::from_hash_of("openmat-figures-dock"),
        egui::ViewportBuilder::default()
            .with_title("Figures")
            .with_inner_size([720.0, 520.0])
            .with_close_button(true),
        |ctx, class| {
            if ctx.input(|input| input.viewport().close_requested()) {
                close_requested = true;
            }
            let mut embedded_open = true;
            let draw = |ui: &mut egui::Ui| {
                ui.horizontal(|ui| {
                    for (index, _) in figures {
                        if ui
                            .selectable_label(*selected == *index, format!("Figure {}", index + 1))
                            .clicked()
                        {
                            *selected = *index;
                        }
                    }
                    if ui.button("Close tab").clicked() {
                        close_tab = Some(*selected);
                    }
                });
                if let Some((_, figure)) = figures.iter().find(|(index, _)| *index == *selected) {
                    draw_figure(ui, *selected, figure);
                }
            };
            if class == egui::ViewportClass::Embedded {
                egui::Window::new("Figures")
                    .id(egui::Id::new("docked-figures"))
                    .open(&mut embedded_open)
                    .resizable(true)
                    .default_size([520.0, 360.0])
                    .show(ctx, draw);
                if !embedded_open {
                    close_requested = true;
                }
            } else {
                egui::CentralPanel::default().show(ctx, draw);
            }
        },
    );
    if close_requested {
        return figures.iter().map(|(index, _)| *index).collect();
    }
    close_tab.into_iter().collect()
}

fn draw_figure(ui: &mut egui::Ui, fig_index: usize, figure: &openmat_lib::Figure) {
    for (index, axes) in figure.axes.iter().enumerate() {
        if axes.series.is_empty() {
            continue;
        }
        let title = if axes.title.is_empty() {
            format!("axes {}", index + 1)
        } else {
            axes.title.clone()
        };
        ui.label(format!("{title}    {}    {}", axes.xlabel, axes.ylabel));
        egui_plot::Plot::new(format!("fig-{fig_index}-axes-{index}"))
            .height(220.0)
            .show(ui, |plot_ui| {
                for series in &axes.series {
                    let points: egui_plot::PlotPoints = series
                        .xs
                        .iter()
                        .zip(series.ys.iter())
                        .map(|(x, y)| [*x, *y])
                        .collect();
                    plot_ui.line(egui_plot::Line::new(points).name(if series.label.is_empty() {
                        "series"
                    } else {
                        series.label.as_str()
                    }));
                }
            });
    }
}
