use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use crate::HilError;

#[derive(Debug, Clone)]
pub struct BoardProfile {
    pub name: String,
    pub fpga: String,
    pub ethernet_host: String,
    pub ethernet_port: u16,
    pub ethernet_proto: String,
    pub jtag_hw_server: String,
    pub jtag_device: String,
    pub registers: BTreeMap<String, u32>,
}

pub fn load_board(name: &str) -> Result<BoardProfile, HilError> {
    let path = find_board_file(name)?;
    let text = fs::read_to_string(&path)
        .map_err(|err| HilError::new(format!("cannot read {}: {err}", path.display())))?;
    parse_board(&text).map_err(|err| HilError::new(format!("{}: {err}", path.display())))
}

fn find_board_file(name: &str) -> Result<PathBuf, HilError> {
    let file = format!("{name}.toml");
    let mut dirs = Vec::new();
    if let Ok(dir) = std::env::var("OPENMAT_BOARDS") {
        dirs.push(PathBuf::from(dir));
    }
    dirs.push(PathBuf::from("boards"));
    dirs.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../boards"));
    for dir in dirs {
        let path = dir.join(&file);
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(HilError::new(format!(
        "no board profile '{file}' in ./boards or OPENMAT_BOARDS"
    )))
}

fn parse_board(text: &str) -> Result<BoardProfile, HilError> {
    let mut section = String::new();
    let mut values: BTreeMap<String, String> = BTreeMap::new();
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].trim().to_string();
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(HilError::new(format!("expected key = value, got '{line}'")));
        };
        let key = if section.is_empty() {
            key.trim().to_string()
        } else {
            format!("{section}.{}", key.trim())
        };
        values.insert(key, unquote(value.trim()));
    }
    let name = values
        .get("name")
        .cloned()
        .ok_or_else(|| HilError::new("missing string field 'name'"))?;
    let port = values
        .get("ethernet.port")
        .map(|s| s.parse::<u16>())
        .transpose()
        .map_err(|_| HilError::new("ethernet.port must be an integer"))?
        .unwrap_or(5000);
    let proto = values
        .get("ethernet.proto")
        .cloned()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "udp".into());
    let mut registers = BTreeMap::new();
    for (key, value) in &values {
        let Some(name) = key.strip_prefix("registers.") else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        let addr = if let Some(hex) = value.strip_prefix("0x") {
            u32::from_str_radix(hex, 16)
        } else {
            value.parse::<u32>()
        }
        .map_err(|_| HilError::new(format!("register '{name}' must be an integer address")))?;
        registers.insert(name.to_string(), addr);
    }
    Ok(BoardProfile {
        name,
        fpga: values.get("fpga").cloned().unwrap_or_default(),
        ethernet_host: values.get("ethernet.host").cloned().unwrap_or_default(),
        ethernet_port: port,
        ethernet_proto: proto,
        jtag_hw_server: values
            .get("jtag.hw_server")
            .cloned()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "localhost:3121".into()),
        jtag_device: values.get("jtag.device").cloned().unwrap_or_default(),
        registers,
    })
}

fn unquote(value: &str) -> String {
    let value = value.trim().trim_end_matches(',');
    if value.len() >= 2 && value.starts_with('"') && value.ends_with('"') {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

