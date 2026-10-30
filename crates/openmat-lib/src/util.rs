use openmat_core::{MatError, MatResult, Matrix, Value};

pub fn need(args: &[Value], n: usize) -> MatResult<()> {
    if args.len() < n {
        Err(MatError::new(format!("expected at least {n} arguments, got {}", args.len())))
    } else {
        Ok(())
    }
}

pub fn matrix_arg(args: &[Value], index: usize) -> MatResult<Matrix> {
    args.get(index)
        .ok_or_else(|| MatError::new("not enough arguments"))?
        .clone()
        .into_matrix()
}

pub fn real_scalar(args: &[Value], index: usize) -> MatResult<f64> {
    args.get(index)
        .ok_or_else(|| MatError::new("not enough arguments"))?
        .real_scalar()
}

pub fn real_vec(value: &Value) -> MatResult<Vec<f64>> {
    Ok(value.clone().into_matrix()?.reals()?)
}

pub fn row(values: Vec<f64>) -> Value {
    Value::Matrix(Matrix::from_reals(1, values.len(), values))
}

pub fn col(values: Vec<f64>) -> Value {
    Value::Matrix(Matrix::from_reals(values.len(), 1, values))
}

pub fn dim_pair(args: &[Value]) -> MatResult<(usize, usize)> {
    if args.len() == 1 {
        let n = positive(args[0].real_scalar()?)?;
        Ok((n, n))
    } else {
        Ok((positive(args[0].real_scalar()?)?, positive(args[1].real_scalar()?)?))
    }
}

pub fn positive(v: f64) -> MatResult<usize> {
    if v < 0.0 || (v - v.round()).abs() > 1e-8 {
        return Err(MatError::new(format!("expected a non-negative integer, got {v}")));
    }
    Ok(v.round() as usize)
}

pub fn char_arg(args: &[Value], index: usize) -> MatResult<String> {
    match args.get(index) {
        Some(Value::Char(s)) => Ok(s.clone()),
        Some(other) => Err(MatError::new(format!(
            "expected a string, got {}",
            other.type_name()
        ))),
        None => Err(MatError::new("not enough arguments")),
    }
}

pub fn empty() -> Value {
    Value::Matrix(Matrix::empty())
}
