use crate::HilError;

pub const MAGIC: &[u8; 4] = b"OMAT";
pub const VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Op {
    Write = 1,
    Read = 2,
    RegWrite = 3,
    RegRead = 4,
    Ack = 5,
    Err = 6,
}

impl Op {
    fn from_u8(v: u8) -> Result<Self, HilError> {
        Ok(match v {
            1 => Op::Write,
            2 => Op::Read,
            3 => Op::RegWrite,
            4 => Op::RegRead,
            5 => Op::Ack,
            6 => Op::Err,
            other => return Err(HilError::new(format!("unknown HIL opcode {other}"))),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub op: Op,
    pub addr: u32,
    pub payload: Vec<u8>,
}

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ 0xedb8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

pub fn encode(frame: &Frame) -> Vec<u8> {
    let mut buf = Vec::with_capacity(18 + frame.payload.len());
    buf.extend_from_slice(MAGIC);
    buf.push(VERSION);
    buf.push(frame.op as u8);
    buf.extend_from_slice(&frame.addr.to_le_bytes());
    buf.extend_from_slice(&(frame.payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(&frame.payload);
    let crc = crc32(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    buf
}

pub fn decode(buf: &[u8]) -> Result<Frame, HilError> {
    if buf.len() < 18 {
        return Err(HilError::new("HIL frame is too short"));
    }
    if &buf[0..4] != MAGIC {
        return Err(HilError::new("HIL frame has a bad magic number"));
    }
    if buf[4] != VERSION {
        return Err(HilError::new("HIL frame has an unsupported version"));
    }
    let op = Op::from_u8(buf[5])?;
    let addr = u32::from_le_bytes(buf[6..10].try_into().unwrap());
    let len = u32::from_le_bytes(buf[10..14].try_into().unwrap()) as usize;
    if buf.len() != 14 + len + 4 {
        return Err(HilError::new("HIL frame length does not match the header"));
    }
    let payload = buf[14..14 + len].to_vec();
    let expected = crc32(&buf[..14 + len]);
    let got = u32::from_le_bytes(buf[14 + len..].try_into().unwrap());
    if expected != got {
        return Err(HilError::new("HIL frame CRC mismatch"));
    }
    Ok(Frame { op, addr, payload })
}

pub fn f64s_to_bytes(samples: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 8);
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

pub fn bytes_to_f64s(bytes: &[u8]) -> Result<Vec<f64>, HilError> {
    if bytes.len() % 8 != 0 {
        return Err(HilError::new("sample payload length is not a multiple of 8"));
    }
    Ok(bytes
        .chunks(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let frame = Frame {
            op: Op::RegWrite,
            addr: 0x10,
            payload: 0xabcdu32.to_le_bytes().to_vec(),
        };
        let bytes = encode(&frame);
        assert_eq!(decode(&bytes).unwrap(), frame);
    }

    #[test]
    fn bad_crc_is_rejected() {
        let frame = Frame {
            op: Op::Read,
            addr: 4,
            payload: Vec::new(),
        };
        let mut bytes = encode(&frame);
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        assert!(decode(&bytes).is_err());
    }
}
