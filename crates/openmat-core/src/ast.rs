#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Ldiv,
    Pow,
    DotMul,
    DotDiv,
    DotLdiv,
    DotPow,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    AndAnd,
    OrOr,
    /// Internal marker. The parser rewrites this into [`Expr::Range`].
    ColonAsRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Plus,
    Minus,
    Not,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Number(f64),
    Imag(f64),
    String(String),
    Ident(String),
    End,
    Colon,
    Unary {
        op: UnOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Range {
        start: Box<Expr>,
        step: Option<Box<Expr>>,
        end: Box<Expr>,
    },
    Matrix {
        rows: Vec<Vec<Expr>>,
    },
    Cell {
        rows: Vec<Vec<Expr>>,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    Field {
        expr: Box<Expr>,
        field: String,
    },
    CellIndex {
        expr: Box<Expr>,
        args: Vec<Expr>,
    },
    Transpose {
        expr: Box<Expr>,
        conjugate: bool,
    },
    Anon {
        params: Vec<String>,
        body: Box<Expr>,
    },
}

#[derive(Debug, Clone)]
pub enum LValue {
    Name(String),
    Index { name: String, args: Vec<Expr> },
    CellIndex { name: String, args: Vec<Expr> },
    Field { name: String, field: String },
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub line: usize,
    pub kind: StmtKind,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    Expr {
        expr: Expr,
        suppress: bool,
    },
    Assign {
        targets: Vec<LValue>,
        expr: Expr,
        suppress: bool,
    },
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        elseifs: Vec<(Expr, Vec<Stmt>)>,
        else_body: Vec<Stmt>,
    },
    For {
        var: String,
        expr: Expr,
        body: Vec<Stmt>,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
    Switch {
        expr: Expr,
        cases: Vec<(Expr, Vec<Stmt>)>,
        otherwise: Vec<Stmt>,
    },
    Break,
    Continue,
    Return,
    Function {
        name: String,
        outputs: Vec<String>,
        params: Vec<String>,
        body: Vec<Stmt>,
    },
}

impl Stmt {
    pub fn new(line: usize, kind: StmtKind) -> Self {
        Self { line, kind }
    }

    pub fn set_suppress(&mut self, suppress: bool) {
        match &mut self.kind {
            StmtKind::Expr { suppress: flag, .. } | StmtKind::Assign { suppress: flag, .. } => {
                *flag = suppress;
            }
            _ => {}
        }
    }
}
