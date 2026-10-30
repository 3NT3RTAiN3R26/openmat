use std::fs;

use openmat_core::{Fi, MatError, MatResult, Session, Value};

use crate::util::{char_arg, matrix_arg, need, positive};

pub fn register(session: &mut Session) {
    session.register_builtin("fi", fi);
    session.register_builtin("fi2hex", fi2hex);
    session.register_builtin("fi_write", fi_write);
    session.register_builtin("double", double);
}

fn fi(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let matrix = matrix_arg(args, 0)?;
    let signed = if args.len() >= 2 {
        args[1].real_scalar()? != 0.0
    } else {
        true
    };
    let word = if args.len() >= 3 {
        positive(args[2].real_scalar()?)? as u32
    } else {
        16
    };
    let frac = if args.len() >= 4 {
        args[3].real_scalar()?.round() as i32
    } else {
        15
    };
    Ok(Value::Fi(Fi::from_matrix(&matrix, signed, word, frac)?))
}

fn fi2hex(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let Value::Fi(fi) = &args[0] else {
        return Err(MatError::new("fi2hex expects an fi value"));
    };
    Ok(Value::Char(fi.hex().join("\n")))
}

fn fi_write(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let Value::Fi(fi) = &args[0] else {
        return Err(MatError::new("fi_write expects an fi value"));
    };
    let path = char_arg(args, 1)?;
    let text = fi.hex().join("\n") + "\n";
    fs::write(&path, text).map_err(|err| MatError::new(format!("cannot write {path}: {err}")))?;
    Ok(crate::util::empty())
}

fn double(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    match &args[0] {
        Value::Fi(fi) => Ok(Value::Matrix(fi.to_matrix())),
        Value::Matrix(m) => Ok(Value::Matrix(m.clone())),
        other => Err(MatError::new(format!(
            "double() does not support {}",
            other.type_name()
        ))),
    }
}
