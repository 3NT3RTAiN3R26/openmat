use crate::ast::{BinOp, Expr, LValue, Stmt, StmtKind, UnOp};
use crate::error::{MatError, MatResult};
use crate::lexer::{lex, Token, TokenKind};

pub fn parse(src: &str) -> MatResult<Vec<Stmt>> {
    let tokens = lex(src)?;
    let mut parser = Parser {
        tokens,
        pos: 0,
        in_matrix: false,
    };
    let stmts = parser.parse_block()?;
    parser.skip_newlines();
    if !parser.is(&TokenKind::Eof) {
        return Err(parser.err("unexpected token after the script"));
    }
    Ok(stmts)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    in_matrix: bool,
}

impl Parser {
    fn parse_block(&mut self) -> MatResult<Vec<Stmt>> {
        let mut stmts = Vec::new();
        loop {
            self.skip_newlines();
            if self.block_boundary() {
                break;
            }
            let mut stmt = self.parse_stmt()?;
            if self.is(&TokenKind::Semicolon) {
                stmt.set_suppress(true);
                self.bump();
            } else if self.is(&TokenKind::Comma) {
                self.bump();
            }
            stmts.push(stmt);
        }
        Ok(stmts)
    }

    fn block_boundary(&self) -> bool {
        self.is(&TokenKind::Eof) || self.is_kw("end") || self.is_kw("else") || self.is_kw("elseif")
            || self.is_kw("case") || self.is_kw("otherwise")
    }

    fn parse_stmt(&mut self) -> MatResult<Stmt> {
        let line = self.current().line;
        let kind = if self.eat_kw("function") {
            self.parse_function()?
        } else if self.eat_kw("if") {
            self.parse_if()?
        } else if self.eat_kw("for") {
            self.parse_for()?
        } else if self.eat_kw("while") {
            self.parse_while()?
        } else if self.eat_kw("switch") {
            self.parse_switch()?
        } else if self.eat_kw("break") {
            StmtKind::Break
        } else if self.eat_kw("continue") {
            StmtKind::Continue
        } else if self.eat_kw("return") {
            StmtKind::Return
        } else if self.is_kw("global") || self.is_kw("persistent") {
            return Err(self.err("global and persistent are not supported"));
        } else {
            let save = self.pos;
            if let Some(targets) = self.try_lvalues() {
                if self.is(&TokenKind::Assign) {
                    self.bump();
                    let expr = self.parse_expr(0)?;
                    StmtKind::Assign {
                        targets,
                        expr,
                        suppress: false,
                    }
                } else {
                    self.pos = save;
                    let expr = self.parse_expr(0)?;
                    StmtKind::Expr {
                        expr,
                        suppress: false,
                    }
                }
            } else {
                self.pos = save;
                let expr = self.parse_expr(0)?;
                StmtKind::Expr {
                    expr,
                    suppress: false,
                }
            }
        };
        Ok(Stmt::new(line, kind))
    }

    fn parse_if(&mut self) -> MatResult<StmtKind> {
        let cond = self.parse_expr(0)?;
        self.skip_separators();
        let then_body = self.parse_block()?;
        let mut elseifs = Vec::new();
        while self.eat_kw("elseif") {
            let c = self.parse_expr(0)?;
            self.skip_separators();
            let body = self.parse_block()?;
            elseifs.push((c, body));
        }
        let else_body = if self.eat_kw("else") {
            self.skip_separators();
            self.parse_block()?
        } else {
            Vec::new()
        };
        self.expect_kw("end")?;
        Ok(StmtKind::If {
            cond,
            then_body,
            elseifs,
            else_body,
        })
    }

    fn parse_for(&mut self) -> MatResult<StmtKind> {
        let var = self.expect_ident()?;
        self.expect(&TokenKind::Assign)?;
        let expr = self.parse_expr(0)?;
        self.skip_separators();
        let body = self.parse_block()?;
        self.expect_kw("end")?;
        Ok(StmtKind::For { var, expr, body })
    }

    fn parse_while(&mut self) -> MatResult<StmtKind> {
        let cond = self.parse_expr(0)?;
        self.skip_separators();
        let body = self.parse_block()?;
        self.expect_kw("end")?;
        Ok(StmtKind::While { cond, body })
    }

    fn parse_switch(&mut self) -> MatResult<StmtKind> {
        let expr = self.parse_expr(0)?;
        self.skip_separators();
        let mut cases = Vec::new();
        while self.eat_kw("case") {
            let c = self.parse_expr(0)?;
            self.skip_separators();
            let body = self.parse_block()?;
            cases.push((c, body));
        }
        let otherwise = if self.eat_kw("otherwise") {
            self.skip_separators();
            self.parse_block()?
        } else {
            Vec::new()
        };
        self.expect_kw("end")?;
        Ok(StmtKind::Switch {
            expr,
            cases,
            otherwise,
        })
    }

    fn parse_function(&mut self) -> MatResult<StmtKind> {
        let (outputs, name, params) = if self.is(&TokenKind::LBracket) {
            let outputs = self.parse_name_list()?;
            self.expect(&TokenKind::Assign)?;
            let name = self.expect_ident()?;
            let params = self.parse_param_list()?;
            (outputs, name, params)
        } else {
            let first = self.expect_ident()?;
            if self.is(&TokenKind::Assign) {
                self.bump();
                let name = self.expect_ident()?;
                let params = self.parse_param_list()?;
                (vec![first], name, params)
            } else {
                let params = self.parse_param_list()?;
                (Vec::new(), first, params)
            }
        };
        self.skip_separators();
        let body = self.parse_block()?;
        self.expect_kw("end")?;
        Ok(StmtKind::Function {
            name,
            outputs,
            params,
            body,
        })
    }

    fn parse_param_list(&mut self) -> MatResult<Vec<String>> {
        if !self.is(&TokenKind::LParen) {
            return Ok(Vec::new());
        }
        self.bump();
        let mut params = Vec::new();
        self.skip_newlines();
        if !self.is(&TokenKind::RParen) {
            loop {
                params.push(self.expect_ident()?);
                self.skip_newlines();
                if self.is(&TokenKind::Comma) {
                    self.bump();
                    self.skip_newlines();
                    continue;
                }
                break;
            }
        }
        self.expect(&TokenKind::RParen)?;
        Ok(params)
    }

    fn parse_name_list(&mut self) -> MatResult<Vec<String>> {
        self.expect(&TokenKind::LBracket)?;
        let mut names = Vec::new();
        loop {
            self.skip_newlines();
            names.push(self.expect_ident()?);
            self.skip_newlines();
            if self.is(&TokenKind::Comma) {
                self.bump();
                continue;
            }
            break;
        }
        self.expect(&TokenKind::RBracket)?;
        Ok(names)
    }

    fn try_lvalues(&mut self) -> Option<Vec<LValue>> {
        let save = self.pos;
        let parsed = self.parse_lvalues();
        match parsed {
            Some(v) => Some(v),
            None => {
                self.pos = save;
                None
            }
        }
    }

    fn parse_lvalues(&mut self) -> Option<Vec<LValue>> {
        if self.is(&TokenKind::LBracket) {
            self.bump();
            let mut names = Vec::new();
            loop {
                self.skip_newlines();
                names.push(LValue::Name(self.expect_ident().ok()?));
                self.skip_newlines();
                if self.is(&TokenKind::Comma) {
                    self.bump();
                    continue;
                }
                break;
            }
            self.expect(&TokenKind::RBracket).ok()?;
            return Some(names);
        }
        let name = self.expect_ident().ok()?;
        if self.is(&TokenKind::LParen) {
            let args = self.parse_arg_list().ok()?;
            return Some(vec![LValue::Index { name, args }]);
        }
        if self.is(&TokenKind::LBrace) {
            let args = self.parse_brace_list().ok()?;
            return Some(vec![LValue::CellIndex { name, args }]);
        }
        if self.is(&TokenKind::Dot) {
            self.bump();
            let field = self.expect_ident().ok()?;
            return Some(vec![LValue::Field { name, field }]);
        }
        Some(vec![LValue::Name(name)])
    }

    fn parse_expr(&mut self, min_prec: u8) -> MatResult<Expr> {
        let mut left = self.parse_unary()?;
        loop {
            if self.in_matrix && self.juxtaposed_sign() {
                break;
            }
            let Some((op, prec, right_assoc)) = self.infix() else {
                break;
            };
            if prec < min_prec {
                break;
            }
            self.bump();
            let next_min = if right_assoc { prec } else { prec + 1 };
            if op == BinOp::ColonAsRange {
                let mid = self.parse_expr(next_min)?;
                if self.is(&TokenKind::Colon) {
                    self.bump();
                    let end = self.parse_expr(next_min)?;
                    left = Expr::Range {
                        start: Box::new(left),
                        step: Some(Box::new(mid)),
                        end: Box::new(end),
                    };
                } else {
                    left = Expr::Range {
                        start: Box::new(left),
                        step: None,
                        end: Box::new(mid),
                    };
                }
                continue;
            }
            let right = self.parse_expr(next_min)?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn juxtaposed_sign(&self) -> bool {
        let Some(tok) = self.tokens.get(self.pos) else {
            return false;
        };
        if !matches!(tok.kind, TokenKind::Plus | TokenKind::Minus) || !tok.preceded_by_space {
            return false;
        }
        match self.tokens.get(self.pos + 1) {
            Some(next) => !next.preceded_by_space && !matches!(next.kind, TokenKind::Newline | TokenKind::Eof),
            None => false,
        }
    }

    fn infix(&self) -> Option<(BinOp, u8, bool)> {
        let kind = &self.tokens.get(self.pos)?.kind;
        let (op, prec, right) = match kind {
            TokenKind::OrOr => (BinOp::OrOr, 1, false),
            TokenKind::AndAnd => (BinOp::AndAnd, 2, false),
            TokenKind::Or => (BinOp::Or, 3, false),
            TokenKind::And => (BinOp::And, 4, false),
            TokenKind::Eq => (BinOp::Eq, 5, false),
            TokenKind::Ne => (BinOp::Ne, 5, false),
            TokenKind::Lt => (BinOp::Lt, 5, false),
            TokenKind::Gt => (BinOp::Gt, 5, false),
            TokenKind::Le => (BinOp::Le, 5, false),
            TokenKind::Ge => (BinOp::Ge, 5, false),
            TokenKind::Colon => (BinOp::ColonAsRange, 6, false),
            TokenKind::Plus => (BinOp::Add, 7, false),
            TokenKind::Minus => (BinOp::Sub, 7, false),
            TokenKind::Mul => (BinOp::Mul, 9, false),
            TokenKind::Div => (BinOp::Div, 9, false),
            TokenKind::Ldiv => (BinOp::Ldiv, 9, false),
            TokenKind::DotMul => (BinOp::DotMul, 9, false),
            TokenKind::DotDiv => (BinOp::DotDiv, 9, false),
            TokenKind::DotLdiv => (BinOp::DotLdiv, 9, false),
            TokenKind::Pow => (BinOp::Pow, 11, false),
            TokenKind::DotPow => (BinOp::DotPow, 11, false),
            _ => return None,
        };
        let _ = right;
        Some((op, prec, false))
    }

    fn parse_unary(&mut self) -> MatResult<Expr> {
        if self.is(&TokenKind::Plus) || self.is(&TokenKind::Minus) || self.is(&TokenKind::Not) {
            let op = match &self.current().kind {
                TokenKind::Plus => UnOp::Plus,
                TokenKind::Minus => UnOp::Minus,
                _ => UnOp::Not,
            };
            self.bump();
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op,
                expr: Box::new(expr),
            });
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> MatResult<Expr> {
        let mut expr = self.parse_primary()?;
        loop {
            if self.is(&TokenKind::LParen) {
                let args = self.parse_arg_list()?;
                expr = Expr::Call {
                    callee: Box::new(expr),
                    args,
                };
            } else if self.is(&TokenKind::LBrace) {
                let args = self.parse_brace_list()?;
                expr = Expr::CellIndex {
                    expr: Box::new(expr),
                    args,
                };
            } else if self.is(&TokenKind::Dot) {
                self.bump();
                let field = self.expect_ident()?;
                expr = Expr::Field {
                    expr: Box::new(expr),
                    field,
                };
            } else if self.is(&TokenKind::Ctrans) {
                self.bump();
                expr = Expr::Transpose {
                    expr: Box::new(expr),
                    conjugate: true,
                };
            } else if self.is(&TokenKind::DotTrans) {
                self.bump();
                expr = Expr::Transpose {
                    expr: Box::new(expr),
                    conjugate: false,
                };
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> MatResult<Expr> {
        let tok = self.current().clone();
        match tok.kind {
            TokenKind::Number(v) => {
                self.bump();
                Ok(Expr::Number(v))
            }
            TokenKind::Imag(v) => {
                self.bump();
                Ok(Expr::Imag(v))
            }
            TokenKind::String(s) => {
                self.bump();
                Ok(Expr::String(s))
            }
            TokenKind::Ident(name) => {
                self.bump();
                if name == "end" {
                    Ok(Expr::End)
                } else {
                    Ok(Expr::Ident(name))
                }
            }
            TokenKind::Colon => {
                self.bump();
                Ok(Expr::Colon)
            }
            TokenKind::LParen => {
                self.bump();
                self.skip_newlines();
                let expr = self.parse_expr(0)?;
                self.skip_newlines();
                self.expect(&TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::LBracket => self.parse_matrix(),
            TokenKind::LBrace => self.parse_cell(),
            TokenKind::At => self.parse_anon(),
            _ => Err(self.err("expected an expression")),
        }
    }

    fn parse_anon(&mut self) -> MatResult<Expr> {
        self.expect(&TokenKind::At)?;
        self.expect(&TokenKind::LParen)?;
        let mut params = Vec::new();
        self.skip_newlines();
        if !self.is(&TokenKind::RParen) {
            loop {
                params.push(self.expect_ident()?);
                self.skip_newlines();
                if self.is(&TokenKind::Comma) {
                    self.bump();
                    self.skip_newlines();
                    continue;
                }
                break;
            }
        }
        self.expect(&TokenKind::RParen)?;
        let body = self.parse_expr(0)?;
        Ok(Expr::Anon {
            params,
            body: Box::new(body),
        })
    }

    fn parse_matrix(&mut self) -> MatResult<Expr> {
        self.expect(&TokenKind::LBracket)?;
        let prev = self.in_matrix;
        self.in_matrix = true;
        let rows = self.parse_delimited_rows(&TokenKind::RBracket)?;
        self.in_matrix = prev;
        self.expect(&TokenKind::RBracket)?;
        Ok(Expr::Matrix { rows })
    }

    fn parse_cell(&mut self) -> MatResult<Expr> {
        self.expect(&TokenKind::LBrace)?;
        let prev = self.in_matrix;
        self.in_matrix = true;
        let rows = self.parse_delimited_rows(&TokenKind::RBrace)?;
        self.in_matrix = prev;
        self.expect(&TokenKind::RBrace)?;
        Ok(Expr::Cell { rows })
    }

    fn parse_delimited_rows(&mut self, end: &TokenKind) -> MatResult<Vec<Vec<Expr>>> {
        let mut rows = Vec::new();
        let mut row = Vec::new();
        loop {
            if self.is(end) {
                break;
            }
            if self.is(&TokenKind::Newline) || self.is(&TokenKind::Semicolon) {
                self.bump();
                if !row.is_empty() {
                    rows.push(std::mem::take(&mut row));
                }
                continue;
            }
            if self.is(&TokenKind::Comma) {
                self.bump();
                continue;
            }
            if self.is(&TokenKind::Eof) {
                return Err(self.err("unterminated matrix"));
            }
            row.push(self.parse_expr(0)?);
        }
        if !row.is_empty() {
            rows.push(row);
        }
        Ok(rows)
    }

    fn parse_arg_list(&mut self) -> MatResult<Vec<Expr>> {
        self.expect(&TokenKind::LParen)?;
        self.parse_list_until(&TokenKind::RParen)
    }

    fn parse_brace_list(&mut self) -> MatResult<Vec<Expr>> {
        self.expect(&TokenKind::LBrace)?;
        self.parse_list_until(&TokenKind::RBrace)
    }

    fn parse_list_until(&mut self, end: &TokenKind) -> MatResult<Vec<Expr>> {
        let mut args = Vec::new();
        self.skip_newlines();
        if !self.is(end) {
            loop {
                self.skip_newlines();
                if self.is(&TokenKind::Colon)
                    && matches!(
                        self.tokens.get(self.pos + 1).map(|t| &t.kind),
                        Some(TokenKind::Comma | TokenKind::RParen | TokenKind::RBrace)
                    )
                {
                    self.bump();
                    args.push(Expr::Colon);
                } else {
                    args.push(self.parse_expr(0)?);
                }
                self.skip_newlines();
                if self.is(&TokenKind::Comma) {
                    self.bump();
                    continue;
                }
                break;
            }
        }
        self.expect(end)?;
        Ok(args)
    }

    fn skip_newlines(&mut self) {
        while self.is(&TokenKind::Newline) {
            self.bump();
        }
    }

    fn skip_separators(&mut self) {
        while self.is(&TokenKind::Newline) || self.is(&TokenKind::Comma) || self.is(&TokenKind::Semicolon) {
            self.bump();
        }
    }

    fn expect_ident(&mut self) -> MatResult<String> {
        match &self.current().kind {
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.bump();
                Ok(name)
            }
            _ => Err(self.err("expected a name")),
        }
    }

    fn expect(&mut self, kind: &TokenKind) -> MatResult<()> {
        if self.is(kind) {
            self.bump();
            Ok(())
        } else {
            Err(self.err(format!("expected {kind:?}")))
        }
    }

    fn expect_kw(&mut self, kw: &str) -> MatResult<()> {
        if self.eat_kw(kw) {
            Ok(())
        } else {
            Err(self.err(format!("expected '{kw}'")))
        }
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        if self.is_kw(kw) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn is_kw(&self, kw: &str) -> bool {
        matches!(&self.current().kind, TokenKind::Ident(name) if name == kw)
    }

    fn is(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.current().kind) == std::mem::discriminant(kind)
    }

    fn current(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn bump(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn err(&self, message: impl Into<String>) -> MatError {
        MatError::new(message).at(self.current().line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix_rows(src: &str) -> Vec<Vec<Expr>> {
        let stmts = parse(src).unwrap();
        match &stmts[0].kind {
            StmtKind::Expr { expr: Expr::Matrix { rows }, .. } => rows.clone(),
            other => panic!("not a matrix: {other:?}"),
        }
    }

    #[test]
    fn juxtaposition_versus_subtraction() {
        let spaced = matrix_rows("[1 -2]");
        assert_eq!(spaced[0].len(), 2);
        let glued = matrix_rows("[1-2]");
        assert_eq!(glued[0].len(), 1);
        let binary = matrix_rows("[1 - 2]");
        assert_eq!(binary[0].len(), 1);
    }
}
