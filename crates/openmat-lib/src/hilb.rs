use std::collections::BTreeMap;

use openmat_core::{MatError, MatResult, Session, Value};
use openmat_hil as hil;

use crate::util::{char_arg, matrix_arg, need, positive, real_scalar};

pub fn register(session: &mut Session) {
    for (name, func) in [
        ("hil.open", open as fn(&mut Session, &[Value]) -> MatResult<Value>),
        ("hil.write", write),
        ("hil.read", read),
        ("hil.regwrite", regwrite),
        ("hil.regread", regread),
        ("hil.close", close),
    ] {
        session.register_builtin(name, func);
    }
}

fn open(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let kind = char_arg(args, 0)?;
    let mut extra = Vec::new();
    for arg in args.iter().skip(1) {
        if let Value::Char(s) = arg {
            extra.push(s.clone());
        } else {
            extra.push(format!("{}", arg.real_scalar()?.round() as i64));
        }
    }
    let id = hil::open(&kind, &extra).map_err(|err| MatError::new(err.message))?;
    let mut map = BTreeMap::new();
    map.insert("id".into(), Value::scalar(id as f64));
    map.insert("transport".into(), Value::Char(kind));
    Ok(Value::Struct(map))
}

fn write(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let id = handle(args)?;
    let samples = matrix_arg(args, 1)?.reals()?;
    hil::write_samples(id, &samples).map_err(|err| MatError::new(err.message))?;
    Ok(crate::util::empty())
}

fn read(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let id = handle(args)?;
    let n = positive(real_scalar(args, 1)?)?;
    let samples = hil::read_samples(id, n).map_err(|err| MatError::new(err.message))?;
    Ok(crate::util::col(samples))
}

fn regwrite(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let id = handle(args)?;
    let addr = real_scalar(args, 1)?.round() as u32;
    let value = real_scalar(args, 2)?.round() as u32;
    hil::reg_write(id, addr, value).map_err(|err| MatError::new(err.message))?;
    Ok(crate::util::empty())
}

fn regread(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let id = handle(args)?;
    let addr = real_scalar(args, 1)?.round() as u32;
    let value = hil::reg_read(id, addr).map_err(|err| MatError::new(err.message))?;
    Ok(Value::scalar(value as f64))
}

fn close(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let id = handle(args)?;
    hil::close(id).map_err(|err| MatError::new(err.message))?;
    Ok(crate::util::empty())
}

fn handle(args: &[Value]) -> MatResult<i64> {
    match &args[0] {
        Value::Struct(map) => Ok(map
            .get("id")
            .ok_or_else(|| MatError::new("hil handle has no id"))?
            .real_scalar()?
            .round() as i64),
        other => Ok(other.real_scalar()?.round() as i64),
    }
}
