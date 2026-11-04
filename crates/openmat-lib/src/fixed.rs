use std::collections::BTreeMap;
use std::fs;

use openmat_core::{Fi, MatError, MatResult, Overflow, PrecMode, Rounding, Session, Value};

use crate::util::{char_arg, matrix_arg, need, positive};

pub fn register(session: &mut Session) {
    session.register_builtin("fi", fi);
    session.register_builtin("sfi", sfi);
    session.register_builtin("ufi", ufi);
    session.register_builtin("fi2hex", fi2hex);
    session.register_builtin("fi_write", fi_write);
    session.register_builtin("double", double);
    session.register_builtin("bin", bin);
    session.register_builtin("hex", hex);
    session.register_builtin("oct", oct);
    session.register_builtin("dec", dec);
    session.register_builtin("storedInteger", stored_integer);
    session.register_builtin("lsb", lsb);
    session.register_builtin("bitget", bitget);
    session.register_builtin("bitset", bitset);
    session.register_builtin("bitsll", bitsll);
    session.register_builtin("bitsrl", bitsrl);
    session.register_builtin("bitsra", bitsra);
    session.register_builtin("numerictype", numerictype);
    session.register_builtin("fimath", fimath);
    session.register_builtin("quantize", quantize);
    session.register_builtin("hex2fi", hex2fi);
    session.register_builtin("fi2bin", bin);
    session.register_builtin("fi2dec", dec);
    session.register_builtin("bitand", bitand);
    session.register_builtin("bitor", bitor);
    session.register_builtin("bitxor", bitxor);
    session.register_builtin("bitshift", bitshift);
    session.register_builtin("bitrol", bitrol);
    session.register_builtin("bitror", bitror);
    session.register_builtin("bitconcat", bitconcat);
    session.register_builtin("bitsliceget", bitsliceget);
    session.register_builtin("reinterpretcast", reinterpretcast);
    session.register_builtin("removefimath", removefimath);
    session.register_builtin("mpy", mpy_fn);
    session.register_builtin("divide", divide_fn);
    session.register_builtin("cordicsin", cordicsin);
    session.register_builtin("cordiccos", cordiccos);
    session.register_builtin("cordicatan2", cordicatan2);
    session.register_builtin("bestprecision", bestprecision);
    session.register_builtin("range", range_fn);
    session.register_builtin("realmax", realmax_fn);
    session.register_builtin("realmin", realmin_fn);
    session.register_builtin("eps", lsb);
    session.register_builtin("prod", prod_fn);
    session.register_builtin("atan2", atan2_fn);
    session.register_builtin("int", stored_integer);
    session.register_builtin("fixed.Quantizer", fixed_quantizer);
    session.register_builtin("quantizer", quantizer_fn);
    session.register_builtin("fixed.add", fixed_add);
    session.register_builtin("fixed.sub", fixed_sub);
    session.register_builtin("mxfi", mxfi);
    session.register_builtin("fixed.autoscale", autoscale);
    session.register_builtin("fixed.interval", fixed_interval);
    session.register_builtin("fixed.lutdivide", lutdivide);
    session.register_builtin("NumericTypeScope", numeric_type_scope);
    session.register_builtin("dfilt.dffir", dffir);
    session.register_builtin("dfilt.dffirt", dffirt);
}

fn is_char(value: &Value) -> bool {
    matches!(value, Value::Char(_))
}

fn fi(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    if args.get(1).is_some_and(is_type_struct) {
        return fi_typed(args);
    }
    cast(args, None)
}

fn sfi(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    cast(args, Some(true))
}

fn ufi(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    cast(args, Some(false))
}

fn cast(args: &[Value], force_signed: Option<bool>) -> MatResult<Value> {
    need(args, 1)?;
    let matrix = matrix_arg(args, 0)?;
    let signed = force_signed.unwrap_or(if args.len() >= 2 && !is_char(&args[1]) {
        args[1].real_scalar()? != 0.0
    } else {
        true
    });
    let numeric_from = if force_signed.is_some() { 1 } else { 2 };
    let word = if args.len() > numeric_from && !is_char(&args[numeric_from]) {
        positive(args[numeric_from].real_scalar()?)? as u32
    } else {
        16
    };
    let frac_at = numeric_from + 1;
    let frac = if args.len() > frac_at && !is_char(&args[frac_at]) {
        args[frac_at].real_scalar()?.round() as i32
    } else {
        15
    };
    let mut rounding = Rounding::Nearest;
    let mut overflow = Overflow::Saturate;
    for arg in args.iter().skip(1) {
        if let Value::Char(text) = arg {
            if let Ok(mode) = Rounding::parse(text) {
                rounding = mode;
            } else {
                overflow = Overflow::parse(text)?;
            }
        }
    }
    Ok(Value::Fi(Fi::from_matrix_mode(
        &matrix, signed, word, frac, rounding, overflow,
    )?))
}

fn expect_fi(args: &[Value]) -> MatResult<&Fi> {
    match args.first() {
        Some(Value::Fi(fi)) => Ok(fi),
        _ => Err(MatError::new("expected an fi value")),
    }
}

fn bin(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let fi = expect_fi(args)?;
    let mask = fi.stored_mask();
    let width = fi.word as usize;
    let text = fi
        .raw
        .iter()
        .map(|raw| format!("{:0width$b}", (*raw as u64) & mask, width = width))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(Value::Char(text))
}

fn hex(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let fi = expect_fi(args)?;
    Ok(Value::Char(fi.hex().join("\n")))
}

fn oct(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let fi = expect_fi(args)?;
    let mask = fi.stored_mask();
    let width = fi.digits(3);
    let text = fi
        .raw
        .iter()
        .map(|raw| format!("{:0width$o}", (*raw as u64) & mask, width = width))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(Value::Char(text))
}

fn dec(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let fi = expect_fi(args)?;
    Ok(Value::Char(
        fi.raw
            .iter()
            .map(|raw| raw.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    ))
}

fn stored_integer(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let fi = expect_fi(args)?;
    let data = fi.raw.iter().copied().map(|raw| raw as f64).collect();
    Ok(Value::Matrix(openmat_core::Matrix::from_reals(
        fi.rows, fi.cols, data,
    )))
}

fn lsb(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let fi = expect_fi(args)?;
    Ok(Value::scalar(2f64.powi(-fi.frac)))
}

fn bit_index(fi: &Fi, args: &[Value]) -> MatResult<u32> {
    need(args, 2)?;
    let bit = args[1].real_scalar()?.round() as i32;
    if bit < 1 || bit as u32 > fi.word {
        return Err(MatError::new("bit position is outside the word"));
    }
    Ok((bit - 1) as u32)
}

fn bitget(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let fi = expect_fi(args)?;
    let shift = bit_index(fi, args)?;
    let data = fi
        .raw
        .iter()
        .map(|raw| if ((*raw as u64) >> shift) & 1 == 1 { 1.0 } else { 0.0 })
        .collect();
    Ok(Value::Matrix(openmat_core::Matrix::from_reals(
        fi.rows, fi.cols, data,
    )))
}

fn with_bits(fi: &Fi, raw: Vec<i64>) -> Fi {
    let mut out = fi.clone();
    let modulus = 1i64 << fi.word;
    out.raw = raw
        .into_iter()
        .map(|value| {
            let mut bits = value.rem_euclid(modulus);
            if fi.signed && bits >= (1i64 << (fi.word - 1)) {
                bits -= modulus;
            }
            bits
        })
        .collect();
    out
}

fn bitset(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let fi = expect_fi(args)?;
    let shift = bit_index(fi, args)?;
    let on = args
        .get(2)
        .map(|v| v.real_scalar().unwrap_or(1.0) != 0.0)
        .unwrap_or(true);
    let raw = fi
        .raw
        .iter()
        .map(|value| {
            let bit = 1i64 << shift;
            if on {
                value | bit
            } else {
                value & !bit
            }
        })
        .collect();
    Ok(Value::Fi(with_bits(fi, raw)))
}

fn bitsll(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let fi = expect_fi(args)?;
    let by = args[1].real_scalar()?.round() as u32;
    let raw = fi.raw.iter().map(|value| value.wrapping_shl(by)).collect();
    Ok(Value::Fi(with_bits(fi, raw)))
}

fn bitsrl(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let fi = expect_fi(args)?;
    let by = args[1].real_scalar()?.round() as u32;
    let mask = fi.stored_mask();
    let raw = fi
        .raw
        .iter()
        .map(|value| (((*value as u64) & mask) >> by) as i64)
        .collect();
    Ok(Value::Fi(with_bits(fi, raw)))
}

fn bitsra(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let fi = expect_fi(args)?;
    let by = args[1].real_scalar()?.round() as u32;
    let raw = fi.raw.iter().map(|value| value >> by).collect();
    Ok(Value::Fi(with_bits(fi, raw)))
}

fn fi2hex(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let fi = expect_fi(args)?;
    Ok(Value::Char(fi.hex().join("\n")))
}

fn fi_write(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let Value::Fi(fi) = &args[0] else {
        return Err(MatError::new("fi_write expects an fi value"));
    };
    let path = char_arg(args, 1)?;
    let kind = if args.len() >= 3 { char_arg(args, 2)? } else { "hex".to_string() };
    let body = match kind.as_str() {
        "hex" => fi.hex().join("\n"),
        "bin" => bits_text(fi, 2),
        "dec" => fi.raw.iter().map(|raw| raw.to_string()).collect::<Vec<_>>().join("\n"),
        _ => return Err(MatError::new("fi_write format must be hex, bin, or dec")),
    };
    let text = format!(
        "signed={} word={} fraction={}\n{body}\n",
        if fi.signed { 1 } else { 0 },
        fi.word,
        fi.frac
    );
    fs::write(&path, text).map_err(|err| MatError::new(format!("cannot write {path}: {err}")))?;
    Ok(crate::util::empty())
}

fn bits_text(fi: &Fi, radix: u32) -> String {
    let mask = fi.stored_mask();
    let width = if radix == 2 { fi.word as usize } else { fi.digits(4) };
    fi.raw
        .iter()
        .map(|raw| {
            let bits = (*raw as u64) & mask;
            if radix == 2 {
                format!("{bits:0width$b}", width = width)
            } else {
                format!("{bits:0width$x}", width = width)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_type_struct(value: &Value) -> bool {
    let Value::Struct(map) = value else {
        return false;
    };
    map.contains_key("WordLength") || map.contains_key("ProductMode") || map.contains_key("RoundingMethod")
}

fn fi_typed(args: &[Value]) -> MatResult<Value> {
    let matrix = matrix_arg(args, 0)?;
    let mut signed = true;
    let mut word = 16u32;
    let mut frac = 15i32;
    let mut rounding = Rounding::Nearest;
    let mut overflow = Overflow::Saturate;
    let mut product_mode = PrecMode::FullPrecision;
    let mut sum_mode = PrecMode::FullPrecision;
    let mut cast_before_sum = true;
    let mut product_word = 32u32;
    let mut product_frac = 30i32;
    let mut sum_word = 32u32;
    let mut sum_frac = 30i32;
    for arg in args.iter().skip(1) {
        let Value::Struct(map) = arg else {
            return Err(MatError::new("fi type arguments must be numerictype or fimath"));
        };
        if let Some(value) = map.get("Signed") {
            signed = value.real_scalar()? != 0.0;
        }
        if let Some(value) = map.get("WordLength") {
            word = positive(value.real_scalar()?)? as u32;
        }
        if let Some(value) = map.get("FractionLength") {
            frac = value.real_scalar()?.round() as i32;
        }
        if let Some(Value::Char(text)) = map.get("RoundingMethod") {
            rounding = Rounding::parse(text)?;
        }
        if let Some(Value::Char(text)) = map.get("OverflowAction") {
            overflow = Overflow::parse(text)?;
        }
        if let Some(Value::Char(text)) = map.get("ProductMode") {
            product_mode = PrecMode::parse(text)?;
        }
        if let Some(Value::Char(text)) = map.get("SumMode") {
            sum_mode = PrecMode::parse(text)?;
        }
        if let Some(value) = map.get("CastBeforeSum") {
            cast_before_sum = value.real_scalar()? != 0.0;
        }
        if let Some(value) = map.get("ProductWordLength") {
            product_word = positive(value.real_scalar()?)? as u32;
        }
        if let Some(value) = map.get("ProductFractionLength") {
            product_frac = value.real_scalar()?.round() as i32;
        }
        if let Some(value) = map.get("SumWordLength") {
            sum_word = positive(value.real_scalar()?)? as u32;
        }
        if let Some(value) = map.get("SumFractionLength") {
            sum_frac = value.real_scalar()?.round() as i32;
        }
    }
    let mut out = Fi::from_matrix_mode(&matrix, signed, word, frac, rounding, overflow)?;
    out.product_mode = product_mode;
    out.sum_mode = sum_mode;
    out.cast_before_sum = cast_before_sum;
    out.product_word = product_word;
    out.product_frac = product_frac;
    out.sum_word = sum_word;
    out.sum_frac = sum_frac;
    Ok(Value::Fi(out))
}

fn char_field(name: &str, text: &str, map: &mut BTreeMap<String, Value>) {
    map.insert(name.to_string(), Value::Char(text.to_string()));
}

fn numerictype(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let signed = if args.is_empty() { 1.0 } else { args[0].real_scalar()? };
    let word = if args.len() >= 2 { positive(args[1].real_scalar()?)? as f64 } else { 16.0 };
    let frac = if args.len() >= 3 { args[2].real_scalar()? } else { 15.0 };
    let mut map = BTreeMap::new();
    map.insert("Signed".into(), Value::logical(signed != 0.0));
    map.insert("WordLength".into(), Value::scalar(word));
    map.insert("FractionLength".into(), Value::scalar(frac));
    map.insert("DataTypeMode".into(), Value::Char("Fixed-point: binary point scaling".into()));
    Ok(Value::Struct(map))
}

fn fimath(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let mut map = BTreeMap::new();
    char_field("RoundingMethod", "Nearest", &mut map);
    char_field("OverflowAction", "Saturate", &mut map);
    char_field("ProductMode", "FullPrecision", &mut map);
    char_field("SumMode", "FullPrecision", &mut map);
    map.insert("CastBeforeSum".into(), Value::logical(true));
    map.insert("ProductWordLength".into(), Value::scalar(32.0));
    map.insert("ProductFractionLength".into(), Value::scalar(30.0));
    map.insert("SumWordLength".into(), Value::scalar(32.0));
    map.insert("SumFractionLength".into(), Value::scalar(30.0));
    let mut i = 0;
    while i + 1 < args.len() {
        let Value::Char(name) = &args[i] else {
            return Err(MatError::new("fimath expects name, value pairs"));
        };
        map.insert(name.clone(), args[i + 1].clone());
        i += 2;
    }
    Ok(Value::Struct(map))
}

fn quantize(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    if args.get(1).is_some_and(|value| matches!(value, Value::Struct(_))) {
        return fi_typed(args);
    }
    fi(&mut Session::new(), args)
}

fn hex2fi(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let text = char_arg(args, 0)?;
    let word = positive(args[1].real_scalar()?)? as u32;
    let frac = args[2].real_scalar()?.round() as i32;
    let signed = if args.len() >= 4 { args[3].real_scalar()? != 0.0 } else { true };
    let bits = u64::from_str_radix(text.trim(), 16).map_err(|_| MatError::new("hex2fi expects hex text"))?;
    let modulus = 1i64 << word;
    let mut raw = (bits as i64).rem_euclid(modulus);
    if signed && raw >= (1i64 << (word - 1)) {
        raw -= modulus;
    }
    let real = raw as f64 / 2f64.powi(frac);
    Fi::from_matrix_mode(&openmat_core::Matrix::scalar(real), signed, word, frac, Rounding::Nearest, Overflow::Saturate)
        .map(Value::Fi)
}

fn same_word(left: &Fi, right: &Fi) -> MatResult<()> {
    if left.word != right.word {
        return Err(MatError::new("bit operations need the same word length"));
    }
    Ok(())
}

fn bit_logic(args: &[Value], op: fn(i64, i64) -> i64) -> MatResult<Value> {
    need(args, 2)?;
    let left = expect_fi(args)?;
    let Value::Fi(right) = &args[1] else {
        return Err(MatError::new("expected an fi value"));
    };
    same_word(left, right)?;
    if left.raw.len() != right.raw.len() {
        return Err(MatError::new("bit operations need the same number of elements"));
    }
    let raw = left.raw.iter().zip(&right.raw).map(|(a, b)| op(*a, *b)).collect();
    Ok(Value::Fi(with_bits(left, raw)))
}

fn bitand(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    bit_logic(args, |a, b| a & b)
}
fn bitor(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    bit_logic(args, |a, b| a | b)
}
fn bitxor(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    bit_logic(args, |a, b| a ^ b)
}

fn bitshift(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let by = args[1].real_scalar()?;
    if by >= 0.0 {
        bitsll(&mut Session::new(), args)
    } else {
        let mut shifted = args.to_vec();
        shifted[1] = Value::scalar(-by);
        bitsra(&mut Session::new(), &shifted)
    }
}

fn rotate(fi: &Fi, by: i32, left: bool) -> Vec<i64> {
    let width = fi.word;
    let mask = fi.stored_mask();
    let by = by.rem_euclid(width as i32) as u32;
    if by == 0 {
        return fi.raw.clone();
    }
    fi.raw
        .iter()
        .map(|value| {
            let bits = (*value as u64) & mask;
            let moved = if left {
                ((bits << by) | (bits >> (width - by))) & mask
            } else {
                ((bits >> by) | (bits << (width - by))) & mask
            };
            moved as i64
        })
        .collect()
}

fn bitrol(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let fi = expect_fi(args)?;
    let by = args[1].real_scalar()?.round() as i32;
    Ok(Value::Fi(with_bits(fi, rotate(fi, by, true))))
}

fn bitror(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let fi = expect_fi(args)?;
    let by = args[1].real_scalar()?.round() as i32;
    Ok(Value::Fi(with_bits(fi, rotate(fi, by, false))))
}

fn bitconcat(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let left = expect_fi(args)?;
    let Value::Fi(right) = &args[1] else {
        return Err(MatError::new("expected an fi value"));
    };
    if left.raw.len() != right.raw.len() {
        return Err(MatError::new("bitconcat needs the same number of elements"));
    }
    let word = left.word + right.word;
    if word > 62 {
        return Err(MatError::new("bitconcat result is wider than 62 bits"));
    }
    let values: Vec<f64> = left
        .raw
        .iter()
        .zip(&right.raw)
        .map(|(a, b)| {
            let hi = (*a as u64) & left.stored_mask();
            let lo = (*b as u64) & right.stored_mask();
            ((hi << right.word) | lo) as f64
        })
        .collect();
    Fi::from_matrix_mode(
        &openmat_core::Matrix::from_reals(left.rows, left.cols, values),
        false,
        word,
        0,
        Rounding::Nearest,
        Overflow::Wrap,
    )
    .map(Value::Fi)
}

fn bitsliceget(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let fi = expect_fi(args)?;
    let upper = args[1].real_scalar()?.round() as i32;
    let lower = args[2].real_scalar()?.round() as i32;
    if lower < 1 || upper < lower || upper as u32 > fi.word {
        return Err(MatError::new("bitsliceget range is outside the word"));
    }
    let width = (upper - lower + 1) as u32;
    let shift = (lower - 1) as u32;
    let mask = (1u64 << width) - 1;
    let values: Vec<f64> = fi
        .raw
        .iter()
        .map(|raw| (((*raw as u64) >> shift) & mask) as f64)
        .collect();
    Fi::from_matrix_mode(
        &openmat_core::Matrix::from_reals(fi.rows, fi.cols, values),
        false,
        width,
        0,
        Rounding::Nearest,
        Overflow::Wrap,
    )
    .map(Value::Fi)
}

fn reinterpretcast(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let fi = expect_fi(args)?.clone();
    let (signed, word, frac) = if let Value::Struct(map) = &args[1] {
        let signed = map.get("Signed").map(|v| v.real_scalar().unwrap_or(1.0) != 0.0).unwrap_or(fi.signed);
        let word = map.get("WordLength").map(|v| v.real_scalar().unwrap_or(fi.word as f64) as u32).unwrap_or(fi.word);
        let frac = map.get("FractionLength").map(|v| v.real_scalar().unwrap_or(fi.frac as f64) as i32).unwrap_or(fi.frac);
        (signed, word, frac)
    } else {
        (args[1].real_scalar()? != 0.0, positive(args[2].real_scalar()?)? as u32, args[3].real_scalar()?.round() as i32)
    };
    if word != fi.word {
        return Err(MatError::new("reinterpretcast keeps the word length"));
    }
    let modulus = 1i64 << word;
    let raw = fi
        .raw
        .iter()
        .map(|value| {
            let mut bits = (*value as u64 & fi.stored_mask()) as i64;
            if signed && bits >= (1i64 << (word - 1)) {
                bits -= modulus;
            }
            if !signed {
                bits = (*value as u64 & fi.stored_mask()) as i64;
            }
            bits
        })
        .collect();
    let mut out = fi;
    out.signed = signed;
    out.frac = frac;
    out.raw = raw;
    Ok(Value::Fi(out))
}

fn removefimath(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let mut fi = expect_fi(args)?.clone();
    fi.rounding = Rounding::Nearest;
    fi.overflow = Overflow::Saturate;
    fi.product_mode = PrecMode::FullPrecision;
    fi.sum_mode = PrecMode::FullPrecision;
    fi.cast_before_sum = true;
    Ok(Value::Fi(fi))
}

fn mpy_fn(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    fixed_bin(args, "dotmul")
}

fn divide_fn(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    fixed_bin(args, "div")
}

fn fixed_bin(args: &[Value], op: &str) -> MatResult<Value> {
    need(args, 2)?;
    let left = match &args[0] {
        Value::Fi(fi) => fi.clone(),
        other => Fi::from_matrix(&other.clone().into_matrix()?, true, 16, 15)?,
    };
    let right = match &args[1] {
        Value::Fi(fi) => fi.clone(),
        other => Fi::from_matrix_mode(&other.clone().into_matrix()?, left.signed, left.word, left.frac, left.rounding, left.overflow)?,
    };
    Fi::binary_op(&left, &right, op)
}

fn fixed_add(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    fixed_bin(args, "add")
}

fn fixed_sub(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    fixed_bin(args, "sub")
}

fn qstep(value: f64, fi: Option<&Fi>) -> f64 {
    let Some(fi) = fi else { return value };
    match Fi::quantize_mode(value, fi.signed, fi.word, fi.frac, fi.rounding, fi.overflow) {
        Ok(raw) => fi.to_f64(raw),
        Err(_) => value,
    }
}

fn cordic_pair(angle: f64, steps: usize, fmt: Option<&Fi>) -> (f64, f64) {
    let mut x = qstep(0.6072529350088814, fmt);
    let mut y = qstep(0.0, fmt);
    let mut z = angle;
    for i in 0..steps {
        let direction = if z >= 0.0 { 1.0 } else { -1.0 };
        let shift = 2f64.powi(-(i as i32));
        let next_x = qstep(x - direction * y * shift, fmt);
        let next_y = qstep(y + direction * x * shift, fmt);
        x = next_x;
        y = next_y;
        z = qstep(z - direction * shift.atan(), fmt);
    }
    (y, x)
}

fn cordic_out(args: &[Value], value: f64) -> MatResult<Value> {
    match args.first() {
        Some(Value::Fi(fi)) => {
            let mut out = Fi::from_matrix_mode(&openmat_core::Matrix::scalar(value), fi.signed, fi.word, fi.frac, fi.rounding, fi.overflow)?;
            out.copy_fimath_from(fi);
            Ok(Value::Fi(out))
        }
        _ => Ok(Value::scalar(value)),
    }
}

fn cordicsin(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let angle = args[0].real_scalar()?;
    let fmt = match args.first() { Some(Value::Fi(fi)) => Some(fi), _ => None };
    let (sine, _) = cordic_pair(angle, 24, fmt);
    cordic_out(args, sine)
}

fn cordiccos(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    let angle = args[0].real_scalar()?;
    let fmt = match args.first() { Some(Value::Fi(fi)) => Some(fi), _ => None };
    let (_, cosine) = cordic_pair(angle, 24, fmt);
    cordic_out(args, cosine)
}

fn cordicatan2(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let fmt = match args.first() { Some(Value::Fi(fi)) => Some(fi), _ => None };
    let mut y = qstep(args[0].real_scalar()?, fmt);
    let mut x = qstep(args[1].real_scalar()?, fmt);
    let mut z = 0.0;
    for i in 0..24 {
        let direction = if y < 0.0 { 1.0 } else { -1.0 };
        let shift = 2f64.powi(-(i as i32));
        let next_x = qstep(x - direction * y * shift, fmt);
        let next_y = qstep(y + direction * x * shift, fmt);
        x = next_x;
        y = next_y;
        z = qstep(z - direction * shift.atan(), fmt);
    }
    cordic_out(args, z)
}

fn bestprecision(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let values = args[0].clone().into_matrix()?.reals()?;
    let word = positive(args[1].real_scalar()?)? as i32;
    let signed = args[2].real_scalar()? != 0.0;
    let max_abs = values.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let magnitude = if max_abs <= 0.0 { 0 } else { max_abs.log2().floor() as i32 + 1 };
    let frac = word - magnitude - if signed { 1 } else { 0 };
    Ok(Value::scalar(frac as f64))
}

fn range_fn(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let fi = expect_fi(args)?;
    let scale = 2f64.powi(fi.frac);
    let (low, high) = if fi.signed {
        (-(1i64 << (fi.word - 1)) as f64 / scale, ((1i64 << (fi.word - 1)) - 1) as f64 / scale)
    } else {
        (0.0, ((1i64 << fi.word) - 1) as f64 / scale)
    };
    let mut out = Fi::from_matrix_mode(&openmat_core::Matrix::from_reals(1, 2, vec![low, high]), fi.signed, fi.word, fi.frac, fi.rounding, fi.overflow)?;
    out.copy_fimath_from(fi);
    Ok(Value::Fi(out))
}

fn realmax_fn(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let Value::Fi(span) = range_fn(&mut Session::new(), args)? else {
        return Err(MatError::new("range did not return fi"));
    };
    Ok(Value::scalar(span.to_f64(span.raw[1])))
}

fn realmin_fn(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    lsb(&mut Session::new(), args)
}

fn fixed_quantizer(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let mut map = BTreeMap::new();
    map.insert("Signed".into(), Value::logical(args[0].real_scalar()? != 0.0));
    map.insert("WordLength".into(), Value::scalar(positive(args[1].real_scalar()?)? as f64));
    map.insert("FractionLength".into(), Value::scalar(args[2].real_scalar()?));
    let rounding = if let Some(Value::Char(text)) = args.get(3) { text.clone() } else { "Nearest".into() };
    let overflow = if let Some(Value::Char(text)) = args.get(4) { text.clone() } else { "Saturate".into() };
    char_field("RoundingMethod", &rounding, &mut map);
    char_field("OverflowAction", &overflow, &mut map);
    Ok(Value::Struct(map))
}

fn quantizer_fn(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let (word, frac) = if args.len() >= 2 && !is_char(&args[0]) {
        (args[0].real_scalar()?, args[1].real_scalar()?)
    } else if args.len() >= 4 {
        let Value::Matrix(m) = &args[3] else {
            return Err(MatError::new("quantizer format is [word fraction]"));
        };
        (m.get(0, 0).re, if m.numel() > 1 { m.get(0, 1).re.max(m.get(1, 0).re) } else { 0.0 })
    } else {
        (16.0, 15.0)
    };
    let mut map = BTreeMap::new();
    map.insert("WordLength".into(), Value::scalar(word));
    map.insert("FractionLength".into(), Value::scalar(frac));
    map.insert("Signed".into(), Value::logical(true));
    char_field("Mode", "fixed", &mut map);
    Ok(Value::Struct(map))
}

fn mxfi(session: &mut Session, _: &[Value]) -> MatResult<Value> {
    let mut map = BTreeMap::new();
    for (name, (low, high)) in &session.fi_seen {
        map.insert(name.clone(), Value::Matrix(openmat_core::Matrix::from_reals(1, 2, vec![*low, *high])));
    }
    Ok(Value::Struct(map))
}

fn autoscale(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 3)?;
    let name = char_arg(args, 0)?;
    let (low, high) = session.fi_seen.get(&name).copied().ok_or_else(|| MatError::new(format!("no fi log for {name}")))?;
    let word = positive(args[1].real_scalar()?)? as i32;
    let signed = args[2].real_scalar()? != 0.0;
    let max_abs = low.abs().max(high.abs());
    let magnitude = if max_abs <= 0.0 { 0 } else { max_abs.log2().floor() as i32 + 1 };
    let frac = word - magnitude - if signed { 1 } else { 0 };
    Ok(Value::scalar(frac as f64))
}

fn lutdivide(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let left = match &args[0] {
        Value::Fi(fi) => fi.clone(),
        other => Fi::from_matrix(&other.clone().into_matrix()?, true, 16, 15)?,
    };
    let right = args[1].clone().into_matrix()?.reals()?;
    let address_bits = if args.len() >= 3 {
        positive(args[2].real_scalar()?)?.clamp(1, 12)
    } else {
        8
    };
    let table: Vec<f64> = (0..(1usize << address_bits))
        .map(|index| {
            let mantissa = 1.0 + (index as f64 + 0.5) / (1u32 << address_bits) as f64;
            1.0 / mantissa
        })
        .collect();
    let numer = left.to_matrix().reals()?;
    if numer.len() != right.len() && right.len() != 1 && numer.len() != 1 {
        return Err(MatError::new("lutdivide inputs must have the same number of elements"));
    }
    let n = numer.len().max(right.len());
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let a = numer[if numer.len() == 1 { 0 } else { i }];
        let b = right[if right.len() == 1 { 0 } else { i }];
        if b == 0.0 {
            return Err(MatError::new("lutdivide divisor is zero"));
        }
        let exponent = b.abs().log2().floor();
        let mantissa = b.abs() / exponent.exp2();
        let index = ((mantissa - 1.0) * (1u32 << address_bits) as f64).floor() as usize;
        let index = index.min(table.len() - 1);
        let quotient = a.signum() * b.signum() * a.abs() * table[index] / exponent.exp2();
        values.push(quotient);
    }
    let rows = if left.rows * left.cols == n { left.rows } else { 1 };
    let cols = if rows == 0 { 0 } else { n / rows };
    let (signed, word, frac) = left.product_format(&left);
    let mut out = Fi::from_matrix_mode(
        &openmat_core::Matrix::from_reals(rows, cols, values),
        signed,
        word,
        frac,
        left.rounding,
        left.overflow,
    )?;
    out.copy_fimath_from(&left);
    Ok(Value::Fi(out))
}

fn numeric_type_scope(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    if let Some(Value::Char(text)) = args.first() {
        if text == "reset" {
            session.numeric_type_scope.clear();
            return Ok(crate::util::empty());
        }
    }
    if let Some(value) = args.first() {
        if !matches!(value, Value::Char(_)) {
            session.numeric_type_scope.extend(value.clone().into_matrix()?.reals()?);
        }
    }
    let samples = session.numeric_type_scope.clone();
    if samples.is_empty() {
        return Err(MatError::new("NumericTypeScope has no samples"));
    }
    let low = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let high = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let bins = 32usize;
    let span = (high - low).max(1e-12);
    let mut counts = vec![0.0; bins];
    for sample in &samples {
        let mut index = (((sample - low) / span) * bins as f64).floor() as usize;
        if index >= bins {
            index = bins - 1;
        }
        counts[index] += 1.0;
    }
    let xs: Vec<f64> = (0..bins).map(|i| low + (i as f64 + 0.5) * span / bins as f64).collect();
    let max_abs = low.abs().max(high.abs());
    let magnitude = if max_abs <= 0.0 { 0 } else { max_abs.log2().floor() as i32 + 1 };
    let frac = 16 - magnitude - 1;
    let figure = session.styled_figure();
    session.figures.push(figure);
    session.current_figure = session.figures.len() - 1;
    let axes = session.figures[session.current_figure].axes_mut();
    axes.series.clear();
    axes.series.push(openmat_core::Series {
        xs,
        ys: counts,
        kind: openmat_core::SeriesKind::Stem,
        label: "histogram".into(),
    });
    axes.title = format!("NumericTypeScope  suggested s16,{frac}");
    axes.xlabel = format!("value  min {low}  max {high}");
    axes.ylabel = "count".into();
    Ok(Value::scalar(frac as f64))
}

fn fixed_interval(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let mut map = BTreeMap::new();
    map.insert("Left".into(), Value::scalar(args[0].real_scalar()?));
    map.insert("Right".into(), Value::scalar(args[1].real_scalar()?));
    Ok(Value::Struct(map))
}

fn dfilt_struct(args: &[Value], form: &str) -> MatResult<Value> {
    need(args, 1)?;
    let mut map = BTreeMap::new();
    map.insert("form".into(), Value::Char(form.to_string()));
    map.insert("b".into(), args[0].clone());
    Ok(Value::Struct(map))
}

fn dffir(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    dfilt_struct(args, "direct")
}

fn dffirt(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    dfilt_struct(args, "transposed")
}

fn prod_fn(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    let fi = expect_fi(args)?;
    let acc = fi.raw.iter().fold(1.0, |acc, raw| acc * fi.to_f64(*raw));
    let mut out = Fi::from_matrix_mode(&openmat_core::Matrix::scalar(acc), fi.signed, fi.word, fi.frac, fi.rounding, fi.overflow)?;
    out.copy_fimath_from(fi);
    Ok(Value::Fi(out))
}

fn atan2_fn(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 2)?;
    let y = args[0].real_scalar()?;
    let x = args[1].real_scalar()?;
    cordic_out(args, y.atan2(x))
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
