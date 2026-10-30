use crate::ast::{Stmt, StmtKind};
use crate::parser::parse;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub line: usize,
    pub message: String,
    pub severity: Severity,
}

#[derive(Clone, Copy)]
enum Frame {
    Block,
    Switch,
}

/// Indents `if` / `for` / `while` / `switch` / `function` bodies by four spaces.
/// Expression text is left unchanged, so `[1 -2]` stays two elements.
pub fn format_source(src: &str) -> String {
    let mut depth: Vec<Frame> = Vec::new();
    let mut case_body = false;
    let mut in_block_comment = false;
    let mut out = String::new();
    for line in split_lines(src) {
        let trimmed = line.trim();
        if in_block_comment {
            push_line(&mut out, 0, line.trim_end());
            if trimmed.contains("%}") {
                in_block_comment = false;
            }
            continue;
        }
        if trimmed.starts_with("%{") {
            push_line(&mut out, indent_of(&depth, case_body), trimmed);
            if !trimmed.contains("%}") {
                in_block_comment = true;
            }
            continue;
        }
        if trimmed.is_empty() {
            out.push('\n');
            continue;
        }
        for piece in statement_pieces(trimmed) {
            let keyword = first_keyword(piece);
            let print_at = match keyword {
                Some("end") => match depth.last().copied() {
                    Some(Frame::Switch) => {
                        case_body = false;
                        depth.pop();
                        depth.len()
                    }
                    Some(Frame::Block) => {
                        depth.pop();
                        indent_of(&depth, case_body)
                    }
                    None => 0,
                },
                Some("else" | "elseif") => depth.len().saturating_sub(1) + usize::from(case_body),
                Some("case" | "otherwise") => {
                    case_body = false;
                    depth.len()
                }
                _ => indent_of(&depth, case_body),
            };
            push_line(&mut out, print_at, piece);
            match keyword {
                Some("if" | "for" | "while" | "function" | "try" | "classdef") => {
                    depth.push(Frame::Block);
                }
                Some("switch") => depth.push(Frame::Switch),
                Some("case" | "otherwise") => case_body = true,
                _ => {}
            }
        }
    }
    if out.ends_with("\n\n") {
        while out.ends_with("\n\n") {
            out.pop();
        }
    }
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

pub fn lint_source(src: &str) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for (index, line) in split_lines(src).into_iter().enumerate() {
        let line_no = index + 1;
        if line.ends_with(' ') || line.ends_with('\t') {
            diagnostics.push(Diagnostic {
                line: line_no,
                message: "trailing whitespace".into(),
                severity: Severity::Warning,
            });
        }
        if line.starts_with('\t') || line.contains('\t') {
            diagnostics.push(Diagnostic {
                line: line_no,
                message: "tab character; the formatter uses spaces".into(),
                severity: Severity::Warning,
            });
        }
        if statement_pieces(line).len() > 1 {
            diagnostics.push(Diagnostic {
                line: line_no,
                message: "more than one statement on this line".into(),
                severity: Severity::Warning,
            });
        }
    }
    match parse(src) {
        Ok(program) => lint_block(&program, false, &mut diagnostics),
        Err(err) => diagnostics.push(Diagnostic {
            line: err.line.unwrap_or(1),
            message: err.message,
            severity: Severity::Error,
        }),
    }
    diagnostics
}

fn lint_block(stmts: &[Stmt], in_loop: bool, diagnostics: &mut Vec<Diagnostic>) {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::Break | StmtKind::Continue if !in_loop => {
                diagnostics.push(Diagnostic {
                    line: stmt.line,
                    message: "break and continue are only valid inside a loop".into(),
                    severity: Severity::Error,
                });
            }
            StmtKind::For { body, .. } | StmtKind::While { body, .. } => {
                lint_block(body, true, diagnostics);
            }
            StmtKind::If {
                then_body,
                elseifs,
                else_body,
                ..
            } => {
                lint_block(then_body, in_loop, diagnostics);
                for (_, body) in elseifs {
                    lint_block(body, in_loop, diagnostics);
                }
                lint_block(else_body, in_loop, diagnostics);
            }
            StmtKind::Switch {
                cases, otherwise, ..
            } => {
                for (_, body) in cases {
                    lint_block(body, in_loop, diagnostics);
                }
                lint_block(otherwise, in_loop, diagnostics);
            }
            StmtKind::Function { body, .. } => lint_block(body, false, diagnostics),
            _ => {}
        }
    }
}

fn split_lines(src: &str) -> Vec<&str> {
    if src.is_empty() {
        return Vec::new();
    }
    src.split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect()
}

/// Splits a physical line at semicolons that separate statements.
/// Semicolons inside `[]`, `()`, `{}`, strings, and comments stay put.
fn statement_pieces(line: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut quote = false;
    let mut comment = false;
    let mut depth = 0i32;
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if comment {
            break;
        }
        if quote {
            if c == '\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                quote = false;
            }
            i += 1;
            continue;
        }
        match c {
            '%' => comment = true,
            '\'' => quote = true,
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ';' if depth == 0 => {
                let piece = line[start..=i].trim();
                if !piece.is_empty() {
                    pieces.push(piece);
                }
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    let rest = line[start..].trim();
    if !rest.is_empty() {
        pieces.push(rest);
    }
    pieces
}

fn first_keyword(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    if trimmed.starts_with('%') {
        return None;
    }
    let end = trimmed
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(trimmed.len());
    if end == 0 {
        return None;
    }
    Some(&trimmed[..end])
}

fn indent_of(depth: &[Frame], case_body: bool) -> usize {
    depth.len() + usize::from(case_body)
}

fn push_line(out: &mut String, level: usize, text: &str) {
    for _ in 0..level {
        out.push_str("    ");
    }
    out.push_str(text);
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indents_if_else_and_switch() {
        let src = "if a\nb = 1\nelse\nb = 2\nend\nswitch x\ncase 1\ny = 1\notherwise\ny = 0\nend\n";
        let formatted = format_source(src);
        assert_eq!(
            formatted,
            "if a\n    b = 1\nelse\n    b = 2\nend\nswitch x\n    case 1\n        y = 1\n    otherwise\n        y = 0\nend\n"
        );
    }

    #[test]
    fn leaves_matrix_spacing_alone() {
        let src = "a = [1 -2];\n";
        assert_eq!(format_source(src), src);
        assert_eq!(format_source("m = [1, 2; 3, 4];\n"), "m = [1, 2; 3, 4];\n");
    }

    #[test]
    fn splits_statements_and_warns_about_semicolons() {
        let src = "hil.write(h, [1 2 3 4]);y = hil.read(h, 4)\nfprintf('v = %d\\n', v)\n";
        assert_eq!(
            format_source(src),
            "hil.write(h, [1 2 3 4]);\ny = hil.read(h, 4)\nfprintf('v = %d\\n', v)\n"
        );
        let diagnostics = lint_source(src);
        assert!(diagnostics
            .iter()
            .any(|d| d.line == 1 && d.message.contains("more than one statement")));
        assert!(diagnostics
            .iter()
            .all(|d| !d.message.contains("semicolon")));
    }

    #[test]
    fn reports_break_outside_a_loop_and_a_syntax_error() {
        let broken = lint_source("break\n");
        assert!(broken.iter().any(|d| d.line == 1 && d.severity == Severity::Error));
        let syntax = lint_source("if a\n");
        assert!(syntax.iter().any(|d| d.severity == Severity::Error));
    }
}
