use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::device::Device;
use crate::protocol::{decode, encode};
use crate::HilError;

pub fn spawn_udp(bind: &str, device: Arc<Mutex<Device>>) -> Result<std::net::SocketAddr, HilError> {
    let socket = UdpSocket::bind(bind).map_err(|err| HilError::new(format!("UDP bind {bind}: {err}")))?;
    socket
        .set_read_timeout(Some(Duration::from_millis(200)))
        .ok();
    let addr = socket
        .local_addr()
        .map_err(|err| HilError::new(err.to_string()))?;
    thread::spawn(move || {
        let mut buf = [0u8; 65535];
        loop {
            let (n, peer) = match socket.recv_from(&mut buf) {
                Ok(pair) => pair,
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock || err.kind() == std::io::ErrorKind::TimedOut => {
                    continue;
                }
                Err(_) => break,
            };
            let response = match decode(&buf[..n]) {
                Ok(frame) => {
                    let mut dev = device.lock().expect("hil device lock");
                    dev.dispatch(frame)
                }
                Err(err) => crate::protocol::Frame {
                    op: crate::protocol::Op::Err,
                    addr: 0,
                    payload: err.to_string().into_bytes(),
                },
            };
            let _ = socket.send_to(&encode(&response), peer);
        }
    });
    Ok(addr)
}

pub fn spawn_tcp(bind: &str, device: Arc<Mutex<Device>>) -> Result<std::net::SocketAddr, HilError> {
    let listener = TcpListener::bind(bind).map_err(|err| HilError::new(format!("TCP bind {bind}: {err}")))?;
    let addr = listener
        .local_addr()
        .map_err(|err| HilError::new(err.to_string()))?;
    thread::spawn(move || {
        for incoming in listener.incoming() {
            let Ok(mut stream) = incoming else { continue };
            let device = Arc::clone(&device);
            thread::spawn(move || {
                let _ = serve_tcp_client(&mut stream, &device);
            });
        }
    });
    Ok(addr)
}

fn serve_tcp_client(stream: &mut TcpStream, device: &Mutex<Device>) -> Result<(), HilError> {
    stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
    loop {
        let mut header = [0u8; 14];
        if stream.read_exact(&mut header).is_err() {
            break;
        }
        let len = u32::from_le_bytes(header[10..14].try_into().unwrap()) as usize;
        let mut rest = vec![0u8; len + 4];
        stream.read_exact(&mut rest).map_err(|err| HilError::new(err.to_string()))?;
        let mut all = header.to_vec();
        all.extend_from_slice(&rest);
        let response = match decode(&all) {
            Ok(frame) => device.lock().expect("hil device lock").dispatch(frame),
            Err(err) => crate::protocol::Frame {
                op: crate::protocol::Op::Err,
                addr: 0,
                payload: err.to_string().into_bytes(),
            },
        };
        stream
            .write_all(&encode(&response))
            .map_err(|err| HilError::new(err.to_string()))?;
    }
    Ok(())
}
