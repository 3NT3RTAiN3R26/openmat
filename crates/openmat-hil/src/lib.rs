mod device;
mod jtag;
mod net;
mod profile;
mod protocol;
mod server;

use std::collections::HashMap;
use std::net::{TcpStream, UdpSocket};
use std::sync::{Mutex, OnceLock};

pub use device::Device;
pub use jtag::{parse_reg_read, vivado_tcl};
pub use profile::{load_board, BoardProfile};
pub use protocol::{decode, encode, Frame, Op};
pub use server::{spawn_tcp, spawn_udp};

#[derive(Debug)]
pub struct HilError {
    pub message: String,
}

impl HilError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for HilError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for HilError {}

enum Conn {
    Local(Device),
    Udp(UdpSocket),
    Tcp(TcpStream),
    Jtag(BoardProfile),
}

fn sessions() -> &'static Mutex<HashMap<i64, Conn>> {
    static SESSIONS: OnceLock<Mutex<HashMap<i64, Conn>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_id() -> i64 {
    static NEXT: OnceLock<Mutex<i64>> = OnceLock::new();
    let next = NEXT.get_or_init(|| Mutex::new(1));
    let mut guard = next.lock().expect("hil id lock");
    let id = *guard;
    *guard += 1;
    id
}

pub fn open(kind: &str, args: &[String]) -> Result<i64, HilError> {
    let conn = match kind {
        "loopback" => Conn::Local(Device::default()),
        "udp" => {
            let host = args.first().map(String::as_str).unwrap_or("127.0.0.1");
            let port = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);
            Conn::Udp(net::udp_socket(host, port)?)
        }
        "tcp" => {
            let host = args.first().map(String::as_str).unwrap_or("127.0.0.1");
            let port = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);
            Conn::Tcp(net::tcp_stream(host, port)?)
        }
        "jtag" => {
            let board = args
                .first()
                .map(String::as_str)
                .unwrap_or("pynq_rfsoc4x2");
            let profile = load_board(board)?;
            if profile.jtag_device.is_empty() {
                return Err(HilError::new(format!(
                    "board '{}' has no JTAG device yet",
                    profile.name
                )));
            }
            Conn::Jtag(profile)
        }
        "jtag-mock" => {
            let board = args
                .first()
                .map(String::as_str)
                .unwrap_or("pynq_rfsoc4x2");
            let _profile = load_board(board)?;
            Conn::Local(Device::default())
        }
        other => {
            return Err(HilError::new(format!(
                "unknown hil transport '{other}'. Use loopback, udp, tcp, jtag, or jtag-mock."
            )))
        }
    };
    let id = next_id();
    sessions().lock().expect("hil session lock").insert(id, conn);
    Ok(id)
}

pub fn close(id: i64) -> Result<(), HilError> {
    sessions()
        .lock()
        .expect("hil session lock")
        .remove(&id)
        .map(|_| ())
        .ok_or_else(|| HilError::new(format!("unknown hil handle {id}")))
}

pub fn write_samples(id: i64, samples: &[f64]) -> Result<(), HilError> {
    let frame = Frame {
        op: Op::Write,
        addr: samples.len() as u32,
        payload: protocol::f64s_to_bytes(samples),
    };
    let _ = transact(id, frame)?;
    Ok(())
}

pub fn read_samples(id: i64, n: usize) -> Result<Vec<f64>, HilError> {
    let frame = Frame {
        op: Op::Read,
        addr: n as u32,
        payload: Vec::new(),
    };
    let response = transact(id, frame)?;
    protocol::bytes_to_f64s(&response.payload)
}

pub fn reg_write(id: i64, addr: u32, value: u32) -> Result<(), HilError> {
    let frame = Frame {
        op: Op::RegWrite,
        addr,
        payload: value.to_le_bytes().to_vec(),
    };
    let _ = transact(id, frame)?;
    Ok(())
}

pub fn reg_read(id: i64, addr: u32) -> Result<u32, HilError> {
    let frame = Frame {
        op: Op::RegRead,
        addr,
        payload: Vec::new(),
    };
    let response = transact(id, frame)?;
    if response.payload.len() != 4 {
        return Err(HilError::new("regread response was not 4 bytes"));
    }
    Ok(u32::from_le_bytes(response.payload.as_slice().try_into().unwrap()))
}

fn transact(id: i64, frame: Frame) -> Result<Frame, HilError> {
    let mut guard = sessions().lock().expect("hil session lock");
    let conn = guard
        .get_mut(&id)
        .ok_or_else(|| HilError::new(format!("unknown hil handle {id}")))?;
    match conn {
        Conn::Local(device) => net::apply_locally(device, frame),
        Conn::Udp(socket) => net::exchange_udp(socket, &frame),
        Conn::Tcp(stream) => net::exchange_tcp(stream, &frame),
        Conn::Jtag(profile) => jtag_transact(profile, &frame),
    }
}

fn jtag_transact(profile: &BoardProfile, frame: &Frame) -> Result<Frame, HilError> {
    match frame.op {
        Op::RegWrite => {
            let value = u32::from_le_bytes(
                frame
                    .payload
                    .as_slice()
                    .try_into()
                    .map_err(|_| HilError::new("regwrite payload must be 4 bytes"))?,
            );
            let tcl = jtag::vivado_tcl(profile, frame.addr, Some(value));
            let _ = jtag::run_vivado(&tcl)?;
            Ok(Frame {
                op: Op::Ack,
                addr: frame.addr,
                payload: Vec::new(),
            })
        }
        Op::RegRead => {
            let tcl = jtag::vivado_tcl(profile, frame.addr, None);
            let text = jtag::run_vivado(&tcl)?;
            let value = jtag::parse_reg_read(&text)?;
            Ok(Frame {
                op: Op::Ack,
                addr: frame.addr,
                payload: value.to_le_bytes().to_vec(),
            })
        }
        Op::Write | Op::Read => Err(HilError::new(
            "JTAG transport supports hil.regread and hil.regwrite. Use Ethernet for sample streams.",
        )),
        Op::Ack | Op::Err => Err(HilError::new("bad JTAG request")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn loopback_registers_and_samples() {
        let id = open("loopback", &[]).unwrap();
        reg_write(id, 4, 42).unwrap();
        assert_eq!(reg_read(id, 4).unwrap(), 42);
        write_samples(id, &[1.0, 2.0, 3.5]).unwrap();
        assert_eq!(read_samples(id, 3).unwrap(), vec![1.0, 2.0, 3.5]);
        close(id).unwrap();
    }

    #[test]
    fn udp_loop_with_agent_thread() {
        let device = Arc::new(std::sync::Mutex::new(Device::default()));
        let addr = spawn_udp("127.0.0.1:0", Arc::clone(&device)).unwrap();
        let id = open("udp", &[addr.ip().to_string(), addr.port().to_string()]).unwrap();
        reg_write(id, 8, 7).unwrap();
        assert_eq!(reg_read(id, 8).unwrap(), 7);
        write_samples(id, &[9.0]).unwrap();
        assert_eq!(read_samples(id, 1).unwrap(), vec![9.0]);
        close(id).unwrap();
    }

    #[test]
    fn jtag_without_vivado_fails_clearly() {
        if std::env::var("OPENMAT_VIVADO").is_ok() || which_vivado() {
            return;
        }
        let id = open("jtag", &["pynq_rfsoc4x2".into()]).unwrap();
        let err = reg_read(id, 0).unwrap_err();
        assert!(err.message.contains("Vivado"), "{}", err.message);
        close(id).unwrap();
    }

    #[test]
    fn jtag_mock_uses_board_profile() {
        let id = open("jtag-mock", &["pynq_rfsoc4x2".into()]).unwrap();
        reg_write(id, 0, 99).unwrap();
        assert_eq!(reg_read(id, 0).unwrap(), 99);
        close(id).unwrap();
    }

    #[test]
    fn explorer_profile_is_a_placeholder() {
        let board = load_board("explorer").unwrap();
        assert!(board.fpga.is_empty() || board.jtag_device.is_empty());
    }

    fn which_vivado() -> bool {
        std::process::Command::new("which")
            .arg("vivado")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}
