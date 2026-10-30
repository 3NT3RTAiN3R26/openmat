use std::collections::{HashMap, VecDeque};

use crate::protocol::{bytes_to_f64s, f64s_to_bytes, Frame, Op};
use crate::HilError;

#[derive(Debug, Default)]
pub struct Device {
    pub regs: HashMap<u32, u32>,
    pub fifo: VecDeque<f64>,
}

impl Device {
    pub fn dispatch(&mut self, frame: Frame) -> Frame {
        match frame.op {
            Op::Write => match bytes_to_f64s(&frame.payload) {
                Ok(samples) => {
                    self.fifo.extend(samples);
                    ack(frame.addr, Vec::new())
                }
                Err(err) => error_frame(err.to_string()),
            },
            Op::Read => {
                let n = frame.addr as usize;
                if self.fifo.len() < n {
                    return error_frame(format!(
                        "only {} samples are buffered, read asked for {n}",
                        self.fifo.len()
                    ));
                }
                let samples: Vec<f64> = self.fifo.drain(0..n).collect();
                ack(n as u32, f64s_to_bytes(&samples))
            }
            Op::RegWrite => {
                if frame.payload.len() != 4 {
                    return error_frame("regwrite payload must be 4 bytes".into());
                }
                let value = u32::from_le_bytes(frame.payload.as_slice().try_into().unwrap());
                self.regs.insert(frame.addr, value);
                ack(frame.addr, Vec::new())
            }
            Op::RegRead => {
                let value = self.regs.get(&frame.addr).copied().unwrap_or(0);
                ack(frame.addr, value.to_le_bytes().to_vec())
            }
            Op::Ack | Op::Err => error_frame("client sent a response opcode".into()),
        }
    }
}

fn ack(addr: u32, payload: Vec<u8>) -> Frame {
    Frame {
        op: Op::Ack,
        addr,
        payload,
    }
}

fn error_frame(message: String) -> Frame {
    Frame {
        op: Op::Err,
        addr: 0,
        payload: message.into_bytes(),
    }
}

pub fn ensure_ack(frame: Frame) -> Result<Frame, HilError> {
    match frame.op {
        Op::Ack => Ok(frame),
        Op::Err => {
            let message = String::from_utf8_lossy(&frame.payload).to_string();
            Err(HilError::new(message))
        }
        _ => Err(HilError::new("HIL peer sent an unexpected opcode")),
    }
}
