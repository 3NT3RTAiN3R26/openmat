use std::collections::{BTreeMap, HashMap};

use num_complex::Complex64;

use crate::ast::{BinOp, Expr, LValue, Stmt, StmtKind, UnOp};
use crate::error::{MatError, MatResult};
use crate::parser::parse;
use crate::session::{Session, UserFunc};
use crate::value::{range_values, FuncHandle, Matrix, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    Next,
    Break,
    Continue,
    Return,
}

impl Session {
    pub fn eval_source(&mut self, src: &str) -> MatResult<()> {
        let program = parse(src)?;
        self.eval_program(&program)
    }

    fn eval_program(&mut self, program: &[Stmt]) -> MatResult<()> {
        for stmt in program {
            if let StmtKind::Function {
                name,
                outputs,
                params,
                body,
            } = &stmt.kind
            {
                self.funcs.insert(
                    name.clone(),
                    UserFunc {
                        params: params.clone(),
                        outputs: outputs.clone(),
                        body: body.clone(),
                    },
                );
            }
        }
        for stmt in program {
            if !matches!(stmt.kind, StmtKind::Function { .. }) {
                let flow = self.exec(stmt)?;
                if flow == Flow::Return {
                    break;
                }
            }
        }
        Ok(())
    }

    fn checkpoint(&mut self, line: usize) -> MatResult<()> {
        let Some(debug) = self.debug.clone() else {
            return Ok(());
        };
        debug.wait(self, line)
    }

    fn exec(&mut self, stmt: &Stmt) -> MatResult<Flow> {
        let line = stmt.line;
        self.checkpoint(line)?;
        match &stmt.kind {
            StmtKind::Expr { expr, suppress } => {
                let value = self.eval(expr)?;
                self.insert("ans".into(), value.clone());
                if !suppress {
                    self.writeln(&format!("ans =\n{value}"));
                }
                Ok(Flow::Next)
            }
            StmtKind::Assign {
                targets,
                expr,
                suppress,
            } => {
                let saved = self.nargout;
                self.nargout = targets.len().max(1);
                let value = self.eval(expr)?;
                self.nargout = saved;
                if targets.len() == 1 {
                    self.assign(&targets[0], value.clone())?;
                    if !suppress {
                        if let LValue::Name(name) = &targets[0] {
                            self.writeln(&format!("{name} =\n{value}"));
                        }
                    }
                } else {
                    let items = match value {
                        Value::Multi(items) => items,
                        other => {
                            return Err(MatError::new(format!(
                                "expected {} outputs, got {}",
                                targets.len(),
                                other.type_name()
                            )));
                        }
                    };
                    if items.len() < targets.len() {
                        return Err(MatError::new(format!(
                            "not enough outputs (got {}, need {})",
                            items.len(),
                            targets.len()
                        )));
                    }
                    for (target, item) in targets.iter().zip(items.into_iter()) {
                        if !suppress {
                            if let LValue::Name(name) = target {
                                self.writeln(&format!("{name} =\n{item}"));
                            }
                        }
                        self.assign(target, item)?;
                    }
                }
                Ok(Flow::Next)
            }
            StmtKind::If {
                cond,
                then_body,
                elseifs,
                else_body,
            } => {
                if self.eval(cond)?.is_true()? {
                    return self.exec_block(then_body);
                }
                for (c, body) in elseifs {
                    if self.eval(c)?.is_true()? {
                        return self.exec_block(body);
                    }
                }
                self.exec_block(else_body)
            }
            StmtKind::For { var, expr, body } => {
                let iter = self.eval(expr)?.into_matrix()?;
                for c in 0..iter.cols.max(if iter.numel() == 0 { 0 } else { iter.cols }) {
                    if iter.cols == 0 {
                        break;
                    }
                    if c > 0 {
                        self.checkpoint(line)?;
                    }
                    self.insert(var.clone(), Value::Matrix(iter.column(c)));
                    match self.exec_block(body)? {
                        Flow::Break => break,
                        Flow::Return => return Ok(Flow::Return),
                        Flow::Continue | Flow::Next => {}
                    }
                }
                Ok(Flow::Next)
            }
            StmtKind::While { cond, body } => {
                let mut guard = 0usize;
                let mut first = true;
                while {
                    if !first {
                        self.checkpoint(line)?;
                    }
                    first = false;
                    self.eval(cond)?.is_true()?
                } {
                    guard += 1;
                    if guard > 1_000_000 {
                        return Err(MatError::new("while loop exceeded 1000000 iterations"));
                    }
                    match self.exec_block(body)? {
                        Flow::Break => break,
                        Flow::Return => return Ok(Flow::Return),
                        Flow::Continue | Flow::Next => {}
                    }
                }
                Ok(Flow::Next)
            }
            StmtKind::Switch {
                expr,
                cases,
                otherwise,
            } => {
                let value = self.eval(expr)?;
                for (case, body) in cases {
                    let cval = self.eval(case)?;
                    if values_equal(&value, &cval)? {
                        return self.exec_block(body);
                    }
                }
                self.exec_block(otherwise)
            }
            StmtKind::Break => Ok(Flow::Break),
            StmtKind::Continue => Ok(Flow::Continue),
            StmtKind::Return => Ok(Flow::Return),
            StmtKind::Function { .. } => Ok(Flow::Next),
        }
    }

    fn exec_block(&mut self, body: &[Stmt]) -> MatResult<Flow> {
        for stmt in body {
            match self.exec(stmt)? {
                Flow::Next => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Next)
    }

    fn assign(&mut self, target: &LValue, value: Value) -> MatResult<()> {
        match target {
            LValue::Name(name) => {
                self.insert(name.clone(), value);
                Ok(())
            }
            LValue::Index { name, args } => {
                let mut base = self
                    .get(name)
                    .cloned()
                    .ok_or_else(|| MatError::new(format!("undefined variable '{name}'")))?;
                let indexed = self.write_index(&mut base, args, value)?;
                let _ = indexed;
                self.insert(name.clone(), base);
                Ok(())
            }
            LValue::CellIndex { name, args } => {
                let mut base = self.get(name).cloned().unwrap_or_else(|| Value::Cell {
                    rows: 0,
                    cols: 0,
                    data: Vec::new(),
                });
                self.write_cell(&mut base, args, value)?;
                self.insert(name.clone(), base);
                Ok(())
            }
            LValue::Field { name, field } => {
                let mut base = self.get(name).cloned().unwrap_or_else(|| Value::Struct(BTreeMap::new()));
                match &mut base {
                    Value::Struct(map) => {
                        map.insert(field.clone(), value);
                    }
                    _ => return Err(MatError::new(format!("'{name}' is not a struct"))),
                }
                self.insert(name.clone(), base);
                Ok(())
            }
        }
    }

    pub fn eval(&mut self, expr: &Expr) -> MatResult<Value> {
        match expr {
            Expr::Number(v) => Ok(Value::scalar(*v)),
            Expr::Imag(v) => Ok(Value::complex(0.0, *v)),
            Expr::String(s) => Ok(Value::Char(s.clone())),
            Expr::Ident(name) => self
                .get(name)
                .cloned()
                .or_else(|| {
                    if self.funcs.contains_key(name) || self.builtins.contains_key(name) {
                        Some(Value::Func(FuncHandle::Named(name.clone())))
                    } else {
                        None
                    }
                })
                .ok_or_else(|| MatError::new(format!("undefined variable '{name}'"))),
            Expr::End => Ok(Value::scalar(self.end_index()?)),
            Expr::Colon => Err(MatError::new("bare ':' is only valid as an index")),
            Expr::Unary { op, expr } => {
                let v = self.eval(expr)?.into_matrix()?;
                let out = match op {
                    UnOp::Plus => v,
                    UnOp::Minus => v.map(|z| -z),
                    UnOp::Not => {
                        let mut m = v.map(|z| {
                            if z.re == 0.0 && z.im == 0.0 {
                                Complex64::from(1.0)
                            } else {
                                Complex64::from(0.0)
                            }
                        });
                        m.logical = true;
                        m
                    }
                };
                Ok(Value::Matrix(out))
            }
            Expr::Binary { op, left, right } => self.eval_binary(*op, left, right),
            Expr::Range { start, step, end } => {
                let a = self.eval(start)?.real_scalar()?;
                let b = self.eval(end)?.real_scalar()?;
                let s = match step {
                    Some(st) => self.eval(st)?.real_scalar()?,
                    None => 1.0,
                };
                let vals = range_values(a, s, b)?;
                Ok(Value::Matrix(Matrix::from_reals(1, vals.len(), vals)))
            }
            Expr::Matrix { rows } => self.eval_matrix(rows),
            Expr::Cell { rows } => self.eval_cell(rows),
            Expr::Call { callee, args } => self.eval_call(callee, args),
            Expr::Field { expr, field } => {
                let base = self.eval(expr)?;
                match base {
                    Value::Struct(map) => map
                        .get(field)
                        .cloned()
                        .ok_or_else(|| MatError::new(format!("no field '{field}'"))),
                    Value::Func(FuncHandle::Named(name)) => {
                        Ok(Value::Func(FuncHandle::Named(format!("{name}.{field}"))))
                    }
                    _ => Err(MatError::new("field reference needs a struct")),
                }
            }
            Expr::CellIndex { expr, args } => {
                let base = self.eval(expr)?;
                self.read_cell(&base, args)
            }
            Expr::Transpose { expr, conjugate } => {
                let m = self.eval(expr)?.into_matrix()?;
                Ok(Value::Matrix(m.transpose(*conjugate)))
            }
            Expr::Anon { params, body } => Ok(Value::Func(FuncHandle::Anon {
                params: params.clone(),
                body: (**body).clone(),
                env: self.lookup_func_env(),
            })),
        }
    }

    fn eval_binary(&mut self, op: BinOp, left: &Expr, right: &Expr) -> MatResult<Value> {
        if op == BinOp::AndAnd || op == BinOp::OrOr {
            let l = self.eval(left)?;
            let lt = l.is_true()?;
            if op == BinOp::AndAnd && !lt {
                return Ok(Value::logical(false));
            }
            if op == BinOp::OrOr && lt {
                return Ok(Value::logical(true));
            }
            let r = self.eval(right)?;
            return Ok(Value::logical(r.is_true()?));
        }
        if op == BinOp::ColonAsRange {
            return Err(MatError::new("internal colon marker leaked into evaluation"));
        }
        let lv = self.eval(left)?;
        let rv = self.eval(right)?;
        if matches!((&lv, &rv), (Value::Fi(_), _) | (_, Value::Fi(_))) {
            return self.fi_binary(op, lv, rv);
        }
        let a = lv.into_matrix()?;
        let b = rv.into_matrix()?;
        let result = match op {
            BinOp::Add => a.ew(&b, |x, y| x + y)?,
            BinOp::Sub => a.ew(&b, |x, y| x - y)?,
            BinOp::Mul => a.matmul(&b)?,
            BinOp::DotMul => a.ew(&b, |x, y| x * y)?,
            BinOp::Div => self.right_divide(&a, &b)?,
            BinOp::DotDiv => a.ew(&b, |x, y| x / y)?,
            BinOp::Ldiv => self.left_divide(&a, &b)?,
            BinOp::DotLdiv => b.ew(&a, |x, y| y / x)?,
            BinOp::Pow => pow_mat(&a, &b)?,
            BinOp::DotPow => a.ew(&b, |x, y| x.powc(y))?,
            BinOp::Eq => cmp_mat(&a, &b, |x, y| x == y)?,
            BinOp::Ne => cmp_mat(&a, &b, |x, y| x != y)?,
            BinOp::Lt => cmp_real(&a, &b, |x, y| x < y)?,
            BinOp::Gt => cmp_real(&a, &b, |x, y| x > y)?,
            BinOp::Le => cmp_real(&a, &b, |x, y| x <= y)?,
            BinOp::Ge => cmp_real(&a, &b, |x, y| x >= y)?,
            BinOp::And => logic_mat(&a, &b, |x, y| x && y)?,
            BinOp::Or => logic_mat(&a, &b, |x, y| x || y)?,
            BinOp::AndAnd | BinOp::OrOr | BinOp::ColonAsRange => unreachable!(),
        };
        Ok(Value::Matrix(result))
    }

    fn fi_binary(&mut self, op: BinOp, left: Value, right: Value) -> MatResult<Value> {
        let (fi_l, fi_r) = (as_fi(&left)?, as_fi(&right)?);
        let format = fi_l.clone();
        let a = fi_l.to_matrix();
        let b = fi_r.to_matrix();
        let numeric = match op {
            BinOp::Add => a.ew(&b, |x, y| x + y)?,
            BinOp::Sub => a.ew(&b, |x, y| x - y)?,
            BinOp::Mul | BinOp::DotMul => a.matmul(&b)?,
            BinOp::Div | BinOp::DotDiv => a.ew(&b, |x, y| x / y)?,
            _ => return Err(MatError::new("that operator is not supported for fi values")),
        };
        Ok(Value::Fi(crate::value::Fi::from_matrix(
            &numeric,
            format.signed,
            format.word,
            format.frac,
        )?))
    }

    fn eval_matrix(&mut self, rows: &[Vec<Expr>]) -> MatResult<Value> {
        let mut built: Vec<Matrix> = Vec::new();
        for row in rows {
            if row.is_empty() {
                continue;
            }
            let mut acc: Option<Matrix> = None;
            for expr in row {
                let m = self.eval(expr)?.into_matrix()?;
                acc = Some(match acc {
                    None => m,
                    Some(a) => a.horzcat(&m)?,
                });
            }
            if let Some(m) = acc {
                built.push(m);
            }
        }
        if built.is_empty() {
            return Ok(Value::Matrix(Matrix::empty()));
        }
        let mut acc = built[0].clone();
        for row in built.iter().skip(1) {
            acc = acc.vertcat(row)?;
        }
        Ok(Value::Matrix(acc))
    }

    fn eval_cell(&mut self, rows: &[Vec<Expr>]) -> MatResult<Value> {
        if rows.is_empty() {
            return Ok(Value::Cell {
                rows: 0,
                cols: 0,
                data: Vec::new(),
            });
        }
        let cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let mut data = Vec::new();
        for row in rows {
            if row.len() != cols {
                return Err(MatError::new("cell rows must have the same number of columns"));
            }
            for expr in row {
                data.push(self.eval(expr)?);
            }
        }
        let nrows = rows.len();
        let mut col_major = vec![Value::Matrix(Matrix::empty()); data.len()];
        for r in 0..nrows {
            for c in 0..cols {
                col_major[c * nrows + r] = data[r * cols + c].clone();
            }
        }
        Ok(Value::Cell {
            rows: nrows,
            cols,
            data: col_major,
        })
    }

    fn eval_call(&mut self, callee: &Expr, args: &[Expr]) -> MatResult<Value> {
        if let Some(name) = callable_name(callee) {
            if self.get(&name).is_none()
                && (self.builtins.contains_key(&name) || self.funcs.contains_key(&name))
            {
                return self.call_named(&name, args);
            }
            if let Expr::Ident(ident) = callee {
                if self.get(ident).is_none() && (self.builtins.contains_key(ident) || self.funcs.contains_key(ident)) {
                    return self.call_named(ident, args);
                }
            }
        }
        if let Expr::Field { expr, field } = callee {
            if let Expr::Ident(name) = expr.as_ref() {
                let qual = format!("{name}.{field}");
                if self.builtins.contains_key(&qual) || self.funcs.contains_key(&qual) {
                    return self.call_named(&qual, args);
                }
            }
        }
        let base = self.eval(callee)?;
        if let Value::Func(handle) = base {
            let values = self.eval_args(args)?;
            return self.call_handle(handle, &values);
        }
        self.index_value(base, args)
    }

    fn eval_args(&mut self, args: &[Expr]) -> MatResult<Vec<Value>> {
        let saved = self.nargout;
        self.nargout = 1;
        let mut values = Vec::with_capacity(args.len());
        for arg in args {
            values.push(self.eval(arg)?);
        }
        self.nargout = saved;
        Ok(values)
    }

    fn call_named(&mut self, name: &str, args: &[Expr]) -> MatResult<Value> {
        let values = self.eval_args(args)?;
        let narg = self.nargout;
        if let Some(func) = self.builtins.get(name).copied() {
            self.nargout = narg;
            let out = func(self, &values)?;
            self.nargout = 1;
            return Ok(out);
        }
        let user = self
            .funcs
            .get(name)
            .cloned()
            .ok_or_else(|| MatError::new(format!("undefined function '{name}'")))?;
        self.call_user(&user, &values)
    }

    fn call_handle(&mut self, handle: FuncHandle, args: &[Value]) -> MatResult<Value> {
        match handle {
            FuncHandle::Named(name) => {
                if let Some(func) = self.builtins.get(&name).copied() {
                    return func(self, args);
                }
                let user = self
                    .funcs
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| MatError::new(format!("undefined function '{name}'")))?;
                self.call_user(&user, args)
            }
            FuncHandle::Anon { params, body, env } => {
                if args.len() != params.len() {
                    return Err(MatError::new(format!(
                        "anonymous function expected {} arguments, got {}",
                        params.len(),
                        args.len()
                    )));
                }
                let mut scope = HashMap::new();
                for (k, v) in env {
                    scope.insert(k, v);
                }
                for (p, a) in params.iter().zip(args.iter()) {
                    scope.insert(p.clone(), a.clone());
                }
                let prev = self.in_func;
                self.in_func = true;
                self.scopes.push(scope);
                let result = self.eval(&body);
                self.scopes.pop();
                self.in_func = prev;
                result
            }
        }
    }

    fn call_user(&mut self, func: &UserFunc, args: &[Value]) -> MatResult<Value> {
        if args.len() != func.params.len() {
            return Err(MatError::new(format!(
                "'{}' expected {} arguments, got {}",
                func.params.len(),
                func.params.len(),
                args.len()
            )));
        }
        let mut scope = HashMap::new();
        for (p, a) in func.params.iter().zip(args.iter()) {
            scope.insert(p.clone(), a.clone());
        }
        for out in &func.outputs {
            scope.insert(out.clone(), Value::Matrix(Matrix::empty()));
        }
        let prev = self.in_func;
        self.in_func = true;
        self.scopes.push(scope);
        let flow = self.exec_block(&func.body);
        let result = match flow {
            Ok(Flow::Return) | Ok(Flow::Next) => {
                if func.outputs.is_empty() {
                    Ok(Value::Matrix(Matrix::empty()))
                } else if func.outputs.len() == 1 {
                    self.get(&func.outputs[0])
                        .cloned()
                        .ok_or_else(|| MatError::new("output was not assigned"))
                } else {
                    let mut items = Vec::new();
                    for name in &func.outputs {
                        items.push(
                            self.get(name)
                                .cloned()
                                .ok_or_else(|| MatError::new(format!("output '{name}' was not assigned")))?,
                        );
                    }
                    Ok(Value::Multi(items))
                }
            }
            Ok(_) => Err(MatError::new("break or continue used outside a loop")),
            Err(e) => Err(e),
        };
        self.scopes.pop();
        self.in_func = prev;
        result
    }

    fn index_value(&mut self, base: Value, args: &[Expr]) -> MatResult<Value> {
        let matrix = base.into_matrix()?;
        let specs = self.eval_indexes(&matrix, args)?;
        Ok(Value::Matrix(apply_index(&matrix, &specs)?))
    }

    fn write_index(&mut self, base: &mut Value, args: &[Expr], value: Value) -> MatResult<()> {
        let matrix = match base {
            Value::Matrix(m) => m.clone(),
            other => {
                return Err(MatError::new(format!(
                    "cannot index {}",
                    other.type_name()
                )))
            }
        };
        let specs = self.eval_indexes(&matrix, args)?;
        let rhs = value.into_matrix()?;
        let mut matrix = matrix;
        write_index(&mut matrix, &specs, &rhs)?;
        *base = Value::Matrix(matrix);
        Ok(())
    }

    fn eval_indexes(&mut self, matrix: &Matrix, args: &[Expr]) -> MatResult<Vec<IndexSpec>> {
        let mut specs = Vec::new();
        for (dim, arg) in args.iter().enumerate() {
            if matches!(arg, Expr::Colon) {
                specs.push(IndexSpec::All);
                continue;
            }
            let value = self.with_index_context(matrix.rows, matrix.cols, args.len(), dim, |session| {
                session.eval(arg)
            })?;
            specs.push(spec_from_value(&value, matrix, args.len(), dim)?);
        }
        Ok(specs)
    }

    fn read_cell(&mut self, base: &Value, args: &[Expr]) -> MatResult<Value> {
        let (rows, cols, data) = match base {
            Value::Cell { rows, cols, data } => (*rows, *cols, data),
            _ => return Err(MatError::new("{} indexing needs a cell array")),
        };
        let dummy = Matrix::zeros(rows, cols);
        let specs = self.eval_indexes(&dummy, args)?;
        let positions = positions(rows, cols, &specs)?;
        if positions.len() != 1 {
            return Err(MatError::new("cell content indexing currently returns one cell"));
        }
        let (r, c) = positions[0];
        Ok(data[c * rows + r].clone())
    }

    fn write_cell(&mut self, base: &mut Value, args: &[Expr], value: Value) -> MatResult<()> {
        if !matches!(base, Value::Cell { .. }) {
            *base = Value::Cell {
                rows: 0,
                cols: 0,
                data: Vec::new(),
            };
        }
        let idx_vals = self.eval_args(args)?;
        if idx_vals.len() == 1 {
            let n = positive_index(idx_vals[0].real_scalar()?)?;
            if let Value::Cell { rows, cols, data } = base {
                let need = n;
                if data.len() < need {
                    let new_len = need;
                    data.resize(new_len, Value::Matrix(Matrix::empty()));
                    *rows = new_len;
                    *cols = 1;
                }
                data[n - 1] = value;
                return Ok(());
            }
        }
        Err(MatError::new("only linear c{k} = value assignment is supported"))
    }
}

fn callable_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Ident(name) => Some(name.clone()),
        Expr::Field { expr, field } => match expr.as_ref() {
            Expr::Ident(name) => Some(format!("{name}.{field}")),
            _ => None,
        },
        _ => None,
    }
}

fn as_fi(value: &Value) -> MatResult<crate::value::Fi> {
    match value {
        Value::Fi(fi) => Ok(fi.clone()),
        Value::Matrix(m) => {
            let fi = crate::value::Fi::from_matrix(m, true, 62, 0)?;
            Ok(fi)
        }
        other => Err(MatError::new(format!(
            "expected fi or double, got {}",
            other.type_name()
        ))),
    }
}

fn values_equal(a: &Value, b: &Value) -> MatResult<bool> {
    match (a, b) {
        (Value::Char(x), Value::Char(y)) => Ok(x == y),
        (Value::Matrix(x), Value::Matrix(y)) => {
            if x.rows != y.rows || x.cols != y.cols {
                return Ok(false);
            }
            Ok(x.data.iter().zip(y.data.iter()).all(|(p, q)| p == q))
        }
        (Value::Matrix(x), Value::Char(y)) | (Value::Char(y), Value::Matrix(x)) => {
            let _ = (x, y);
            Ok(false)
        }
        _ => Ok(false),
    }
}

impl Session {
    fn right_divide(&self, a: &Matrix, b: &Matrix) -> MatResult<Matrix> {
        if b.is_scalar() {
            let d = b.data[0];
            return Ok(a.map(|x| x / d));
        }
        if let Some(hook) = self.rdiv_hook {
            return hook(a, b);
        }
        Err(MatError::new("matrix right division is not available"))
    }

    fn left_divide(&self, a: &Matrix, b: &Matrix) -> MatResult<Matrix> {
        if a.is_scalar() {
            let d = a.data[0];
            return Ok(b.map(|x| x / d));
        }
        if let Some(hook) = self.ldiv_hook {
            return hook(a, b);
        }
        Err(MatError::new("matrix left division is not available"))
    }
}

fn pow_mat(a: &Matrix, b: &Matrix) -> MatResult<Matrix> {
    if a.is_scalar() && b.is_scalar() {
        return Ok(Matrix {
            rows: 1,
            cols: 1,
            data: vec![a.data[0].powc(b.data[0])],
            logical: false,
        });
    }
    Err(MatError::new("matrix power is only implemented for scalars; use .^ for elements"))
}

fn cmp_mat(a: &Matrix, b: &Matrix, pred: impl Fn(Complex64, Complex64) -> bool) -> MatResult<Matrix> {
    let m = a.ew(b, |x, y| Complex64::from(if pred(x, y) { 1.0 } else { 0.0 }))?;
    let mut m = m;
    m.logical = true;
    Ok(m)
}

fn cmp_real(a: &Matrix, b: &Matrix, pred: impl Fn(f64, f64) -> bool) -> MatResult<Matrix> {
    if !a.is_real() || !b.is_real() {
        return Err(MatError::new("relational operators need real values"));
    }
    cmp_mat(a, b, |x, y| pred(x.re, y.re))
}

fn logic_mat(a: &Matrix, b: &Matrix, pred: impl Fn(bool, bool) -> bool) -> MatResult<Matrix> {
    cmp_mat(a, b, |x, y| {
        pred(!(x.re == 0.0 && x.im == 0.0), !(y.re == 0.0 && y.im == 0.0))
    })
}

#[derive(Debug)]
enum IndexSpec {
    All,
    Elems(Vec<usize>),
}

fn spec_from_value(value: &Value, matrix: &Matrix, nidx: usize, dim: usize) -> MatResult<IndexSpec> {
    let m = value.as_matrix()?;
    if m.logical {
        if nidx != 1 {
            return Err(MatError::new("logical indexing uses one index"));
        }
        let mut elems = Vec::new();
        for (i, z) in m.data.iter().enumerate() {
            if z.re != 0.0 {
                elems.push(i);
            }
        }
        return Ok(IndexSpec::Elems(elems));
    }
    let mut elems = Vec::new();
    for z in &m.data {
        if z.im.abs() > 1e-12 {
            return Err(MatError::new("index must be real"));
        }
        let n = positive_index(z.re)?;
        let limit = if nidx == 1 {
            matrix.numel()
        } else if dim == 0 {
            matrix.rows
        } else {
            matrix.cols
        };
        if n > limit {
            return Err(MatError::new(format!("index {n} exceeds dimension {limit}")));
        }
        elems.push(n - 1);
    }
    Ok(IndexSpec::Elems(elems))
}

fn positive_index(v: f64) -> MatResult<usize> {
    if (v - v.round()).abs() > 1e-8 || v < 1.0 {
        return Err(MatError::new(format!("index must be a positive integer, got {v}")));
    }
    Ok(v.round() as usize)
}

fn apply_index(matrix: &Matrix, specs: &[IndexSpec]) -> MatResult<Matrix> {
    if specs.len() == 1 {
        let elems = expand_one(matrix, &specs[0])?;
        let mut out = Matrix::zeros(elems.len(), 1);
        for (i, (r, c)) in elems.iter().enumerate() {
            out.set(i, 0, matrix.get(*r, *c));
        }
        if matrix.cols == 1 || matrix.rows == 1 {
            if matrix.rows == 1 {
                out = out.transpose(false);
            }
        }
        return Ok(out);
    }
    if specs.len() != 2 {
        return Err(MatError::new("only one or two indexes are supported"));
    }
    let rows = expand_dim(matrix.rows, &specs[0])?;
    let cols = expand_dim(matrix.cols, &specs[1])?;
    let mut out = Matrix::zeros(rows.len(), cols.len());
    for (ci, c) in cols.iter().enumerate() {
        for (ri, r) in rows.iter().enumerate() {
            out.set(ri, ci, matrix.get(*r, *c));
        }
    }
    Ok(out)
}

fn expand_one(matrix: &Matrix, spec: &IndexSpec) -> MatResult<Vec<(usize, usize)>> {
    match spec {
        IndexSpec::All => Ok((0..matrix.numel())
            .map(|i| (i % matrix.rows, i / matrix.rows))
            .collect()),
        IndexSpec::Elems(elems) => Ok(elems
            .iter()
            .map(|i| (i % matrix.rows, i / matrix.rows))
            .collect()),
    }
}

fn expand_dim(len: usize, spec: &IndexSpec) -> MatResult<Vec<usize>> {
    match spec {
        IndexSpec::All => Ok((0..len).collect()),
        IndexSpec::Elems(elems) => Ok(elems.clone()),
    }
}

fn positions(rows: usize, cols: usize, specs: &[IndexSpec]) -> MatResult<Vec<(usize, usize)>> {
    let dummy = Matrix::zeros(rows, cols);
    let indexed = apply_index(&dummy, specs)?;
    let mut out = Vec::new();
    for c in 0..indexed.cols {
        for r in 0..indexed.rows {
            let z = indexed.get(r, c);
            let _ = z;
        }
    }
    if specs.len() == 1 {
        return expand_one(&Matrix::zeros(rows, cols), &specs[0]);
    }
    let rs = expand_dim(rows, &specs[0])?;
    let cs = expand_dim(cols, &specs[1])?;
    for c in cs {
        for r in &rs {
            out.push((*r, c));
        }
    }
    Ok(out)
}

fn write_index(matrix: &mut Matrix, specs: &[IndexSpec], rhs: &Matrix) -> MatResult<()> {
    let pos = if specs.len() == 1 {
        expand_one(matrix, &specs[0])?
    } else if specs.len() == 2 {
        let rs = expand_dim(matrix.rows, &specs[0])?;
        let cs = expand_dim(matrix.cols, &specs[1])?;
        let mut pairs = Vec::new();
        for c in cs {
            for r in &rs {
                pairs.push((*r, c));
            }
        }
        pairs
    } else {
        return Err(MatError::new("only one or two indexes are supported"));
    };
    if rhs.is_scalar() {
        for (r, c) in pos {
            matrix.set(r, c, rhs.data[0]);
        }
        return Ok(());
    }
    if rhs.numel() != pos.len() {
        return Err(MatError::new("the number of values does not match the index"));
    }
    for (i, (r, c)) in pos.iter().enumerate() {
        let rr = i % rhs.rows;
        let cc = i / rhs.rows;
        matrix.set(*r, *c, rhs.get(rr, cc));
    }
    Ok(())
}
