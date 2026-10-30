use openmat_core::{format_number, MatError, MatResult, Session, Value};

use crate::util::{char_arg, need};

pub fn register(session: &mut Session) {
    session.register_builtin("disp", disp);
    session.register_builtin("fprintf", fprintf);
    session.register_builtin("sprintf", sprintf);
    session.register_builtin("printf", fprintf);
}

fn disp(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    need(args, 1)?;
    session.writeln(&args[0].to_string());
    Ok(crate::util::empty())
}

fn fprintf(session: &mut Session, args: &[Value]) -> MatResult<Value> {
    let text = format_args(args)?;
    session.write_raw(&text);
    Ok(crate::util::empty())
}

fn sprintf(_: &mut Session, args: &[Value]) -> MatResult<Value> {
    Ok(Value::Char(format_args(args)?))
}

fn format_args(args: &[Value]) -> MatResult<String> {
    need(args, 1)?;
    let (fmt_index, first_value) = if matches!(args[0], Value::Char(_)) {
        (0, 1)
    } else if args.len() >= 2 && matches!(args[1], Value::Char(_)) {
        (1, 2)
    } else {
        return Err(MatError::new("fprintf needs a format string"));
    };
    let fmt = char_arg(args, fmt_index)?;
    render(&fmt, &args[first_value..])
}

fn render(fmt: &str, values: &[Value]) -> MatResult<String> {
    let chars: Vec<char> = fmt.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    let mut arg = 0;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() {
            match chars[i + 1] {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                '\\' => out.push('\\'),
                other => {
                    out.push('\\');
                    out.push(other);
                }
            }
            i += 2;
            continue;
        }
        if chars[i] == '%' && i + 1 < chars.len() {
            if chars[i + 1] == '%' {
                out.push('%');
                i += 2;
                continue;
            }
            let spec = chars[i + 1];
            let value = values.get(arg).ok_or_else(|| MatError::new("not enough fprintf arguments"))?;
            arg += 1;
            match spec {
                'd' | 'i' => out.push_str(&format!("{}", value.real_scalar()?.round() as i64)),
                'f' => out.push_str(&format!("{:.6}", value.real_scalar()?)),
                'g' => out.push_str(&format_number(value.real_scalar()?)),
                's' => match value {
                    Value::Char(s) => out.push_str(s),
                    other => out.push_str(&other.to_string()),
                },
                other => return Err(MatError::new(format!("unsupported format %{other}"))),
            }
            i += 2;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    Ok(out)
}
