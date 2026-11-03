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
    match args.len() {
        1 => {
            let n = positive(args[0].real_scalar()?)?;
            Ok((n, n))
        }
        2 => Ok((
            positive(args[0].real_scalar()?)?,
            positive(args[1].real_scalar()?)?,
        )),
        n => Err(MatError::new(format!(
            "expected 1 or 2 dimension arguments, got {n}"
        ))),
    }
}

pub fn positive(v: f64) -> MatResult<usize> {
    if !v.is_finite() || v < 0.0 || (v - v.round()).abs() > 1e-8 || v >= usize::MAX as f64 + 1.0 {
        return Err(MatError::new(format!(
            "expected a non-negative integer, got {v}"
        )));
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

#[cfg(test)]
mod positive_tests {
    use super::positive;

    #[test]
    fn rejects_the_exclusive_usize_upper_bound() {
        assert!(positive(18446744073709551616.0).is_err() || usize::BITS < 64);
        assert!(positive(usize::MAX as f64 + 1.0).is_err());
        assert_eq!(positive(3.0).unwrap(), 3);
    }
}
