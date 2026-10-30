use crate::error::{MatError, MatResult};

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub line: usize,
    pub preceded_by_space: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Number(f64),
    Imag(f64),
    String(String),
    Ident(String),
    Plus,
    Minus,
    Mul,
    Div,
    Ldiv,
    Pow,
    DotMul,
    DotDiv,
    DotLdiv,
    DotPow,
    DotTrans,
    Dot,
    Ctrans,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Assign,
    And,
    Or,
    AndAnd,
    OrOr,
    Not,
    Colon,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Semicolon,
    Newline,
    At,
    Eof,
}

pub fn lex(src: &str) -> MatResult<Vec<Token>> {
    Lexer::new(src).tokenize()
}

struct Lexer<'a> {
    src: &'a [u8],
    i: usize,
    line: usize,
    last: Option<TokenKind>,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src: src.as_bytes(),
            i: 0,
            line: 1,
            last: None,
        }
    }

    fn tokenize(mut self) -> MatResult<Vec<Token>> {
        let mut out = Vec::new();
        while self.i < self.src.len() {
            let spaced = self.skip_space_and_comments()?;
            if self.i >= self.src.len() {
                break;
            }
            let line = self.line;
            let kind = self.next_kind()?;
            self.last = Some(kind.clone());
            out.push(Token {
                kind,
                line,
                preceded_by_space: spaced,
            });
        }
        out.push(Token {
            kind: TokenKind::Eof,
            line: self.line,
            preceded_by_space: false,
        });
        Ok(out)
    }

    fn skip_space_and_comments(&mut self) -> MatResult<bool> {
        let mut spaced = false;
        loop {
            if self.starts_with(b"%{") {
                self.i += 2;
                let end = self.find_bytes(b"%}");
                let end = end.ok_or_else(|| MatError::new("unterminated block comment").at(self.line))?;
                let chunk = &self.src[self.i..end];
                self.line += chunk.iter().filter(|b| **b == b'\n').count();
                self.i = end + 2;
                spaced = true;
                continue;
            }
            if self.peek() == Some(b'%') {
                while let Some(c) = self.peek() {
                    if c == b'\n' {
                        break;
                    }
                    self.i += 1;
                }
                continue;
            }
            if self.starts_with(b"...") {
                self.i += 3;
                while matches!(self.peek(), Some(b' ' | b'\t' | b'\r')) {
                    self.i += 1;
                }
                if self.peek() == Some(b'\n') {
                    self.i += 1;
                    self.line += 1;
                }
                spaced = true;
                continue;
            }
            match self.peek() {
                Some(b' ' | b'\t' | b'\r') => {
                    self.i += 1;
                    spaced = true;
                }
                _ => break,
            }
        }
        Ok(spaced)
    }

    fn next_kind(&mut self) -> MatResult<TokenKind> {
        let c = self.peek().unwrap();
        if c == b'\n' {
            self.i += 1;
            self.line += 1;
            return Ok(TokenKind::Newline);
        }
        if let Some(kind) = self.two_char() {
            return Ok(kind);
        }
        if c == b'\'' {
            if self.can_start_string() {
                return self.string(b'\'');
            }
            self.i += 1;
            return Ok(TokenKind::Ctrans);
        }
        if c == b'"' {
            return self.string(b'"');
        }
        if c.is_ascii_digit() || (c == b'.' && self.peek_at(1).is_some_and(|d| d.is_ascii_digit())) {
            return self.number();
        }
        if c == b'.' {
            self.i += 1;
            return Ok(TokenKind::Dot);
        }
        if is_ident_start(c) {
            return Ok(self.ident());
        }
        self.i += 1;
        let kind = match c {
            b'+' => TokenKind::Plus,
            b'-' => TokenKind::Minus,
            b'*' => TokenKind::Mul,
            b'/' => TokenKind::Div,
            b'\\' => TokenKind::Ldiv,
            b'^' => TokenKind::Pow,
            b'=' => TokenKind::Assign,
            b'<' => TokenKind::Lt,
            b'>' => TokenKind::Gt,
            b'&' => TokenKind::And,
            b'|' => TokenKind::Or,
            b'~' => TokenKind::Not,
            b':' => TokenKind::Colon,
            b'(' => TokenKind::LParen,
            b')' => TokenKind::RParen,
            b'[' => TokenKind::LBracket,
            b']' => TokenKind::RBracket,
            b'{' => TokenKind::LBrace,
            b'}' => TokenKind::RBrace,
            b',' => TokenKind::Comma,
            b';' => TokenKind::Semicolon,
            b'@' => TokenKind::At,
            _ => {
                return Err(MatError::new(format!("unexpected character '{}'", c as char)).at(self.line));
            }
        };
        Ok(kind)
    }

    fn two_char(&mut self) -> Option<TokenKind> {
        let pair = [self.peek()?, self.peek_at(1)?];
        let kind = match &pair {
            b".*" => TokenKind::DotMul,
            b"./" => TokenKind::DotDiv,
            b".\\" => TokenKind::DotLdiv,
            b".^" => TokenKind::DotPow,
            b".'" => TokenKind::DotTrans,
            b"==" => TokenKind::Eq,
            b"~=" => TokenKind::Ne,
            b"<=" => TokenKind::Le,
            b">=" => TokenKind::Ge,
            b"&&" => TokenKind::AndAnd,
            b"||" => TokenKind::OrOr,
            _ => return None,
        };
        self.i += 2;
        Some(kind)
    }

    fn number(&mut self) -> MatResult<TokenKind> {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        if self.peek() == Some(b'.') && self.peek_at(1) != Some(b'*') && self.peek_at(1) != Some(b'/')
            && self.peek_at(1) != Some(b'\\') && self.peek_at(1) != Some(b'^') && self.peek_at(1) != Some(b'\'')
        {
            self.i += 1;
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            let e = self.i;
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.i += 1;
                }
            } else {
                self.i = e;
            }
        }
        let text = std::str::from_utf8(&self.src[start..self.i]).unwrap();
        let value: f64 = text
            .parse()
            .map_err(|_| MatError::new(format!("invalid number '{text}'")).at(self.line))?;
        if matches!(self.peek(), Some(b'i' | b'j')) && !self.peek_at(1).is_some_and(is_ident_continue) {
            self.i += 1;
            return Ok(TokenKind::Imag(value));
        }
        Ok(TokenKind::Number(value))
    }

    fn string(&mut self, quote: u8) -> MatResult<TokenKind> {
        self.i += 1;
        let mut out = String::new();
        while let Some(c) = self.peek() {
            if c == b'\n' {
                return Err(MatError::new("unterminated string").at(self.line));
            }
            if c == quote {
                if self.peek_at(1) == Some(quote) {
                    out.push(quote as char);
                    self.i += 2;
                    continue;
                }
                self.i += 1;
                return Ok(TokenKind::String(out));
            }
            out.push(c as char);
            self.i += 1;
        }
        Err(MatError::new("unterminated string").at(self.line))
    }

    fn ident(&mut self) -> TokenKind {
        let start = self.i;
        self.i += 1;
        while self.peek().is_some_and(is_ident_continue) {
            self.i += 1;
        }
        let text = std::str::from_utf8(&self.src[start..self.i]).unwrap().to_string();
        TokenKind::Ident(text)
    }

    fn can_start_string(&self) -> bool {
        !matches!(
            self.last,
            Some(
                TokenKind::Number(_)
                    | TokenKind::Imag(_)
                    | TokenKind::Ident(_)
                    | TokenKind::RParen
                    | TokenKind::RBracket
                    | TokenKind::RBrace
                    | TokenKind::String(_)
                    | TokenKind::Ctrans
                    | TokenKind::DotTrans
            )
        )
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.i).copied()
    }

    fn peek_at(&self, n: usize) -> Option<u8> {
        self.src.get(self.i + n).copied()
    }

    fn starts_with(&self, bytes: &[u8]) -> bool {
        self.src[self.i..].starts_with(bytes)
    }

    fn find_bytes(&self, pat: &[u8]) -> Option<usize> {
        self.src[self.i..]
            .windows(pat.len())
            .position(|w| w == pat)
            .map(|p| self.i + p)
    }
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_ident_continue(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_versus_transpose() {
        let tokens = lex("a' + 'hi'").unwrap();
        assert!(matches!(tokens[0].kind, TokenKind::Ident(_)));
        assert!(matches!(tokens[1].kind, TokenKind::Ctrans));
        assert!(matches!(tokens[3].kind, TokenKind::String(ref s) if s == "hi"));
    }

    #[test]
    fn imaginary_suffix() {
        let tokens = lex("2i").unwrap();
        assert!(matches!(tokens[0].kind, TokenKind::Imag(v) if v == 2.0));
    }
}
