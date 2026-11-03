use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::device::Device;
use crate::protocol::{decode, encode, payload_len, MAX_TCP_CLIENTS};
use crate::HilError;

/// Returns true when this connection may proceed. The caller must `release_tcp_client` later.
pub fn try_admit_tcp_client(active: &AtomicUsize) -> bool {
    let mut current = active.load(Ordering::Relaxed);
    loop {
        if current >= MAX_TCP_CLIENTS {
            return false;
        }
        match active.compare_exchange_weak(
            current,
            current + 1,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => return true,
            Err(observed) => current = observed,
        }
    }
}

pub fn release_tcp_client(active: &AtomicUsize) {
    active.fetch_sub(1, Ordering::AcqRel);
}

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
    let active = Arc::new(AtomicUsize::new(0));
    thread::spawn(move || {
        for incoming in listener.incoming() {
            let Ok(mut stream) = incoming else { continue };
            if !try_admit_tcp_client(&active) {
                let _ = stream.shutdown(std::net::Shutdown::Both);
                continue;
            }
            let device = Arc::clone(&device);
            let active = Arc::clone(&active);
            thread::spawn(move || {
                let _ = serve_tcp_client(&mut stream, &device);
                release_tcp_client(&active);
            });
        }
    });
    Ok(addr)
}

fn serve_tcp_client(stream: &mut TcpStream, device: &Mutex<Device>) -> Result<(), HilError> {
    stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(30))).ok();
    loop {
        let mut header = [0u8; 14];
        if stream.read_exact(&mut header).is_err() {
            break;
        }
        let len = match payload_len(&header) {
            Ok(len) => len,
            Err(err) => {
                let response = crate::protocol::Frame {
                    op: crate::protocol::Op::Err,
                    addr: 0,
                    payload: err.to_string().into_bytes(),
                };
                let _ = stream.write_all(&encode(&response));
                break;
            }
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{payload_len, MAX_FRAME_PAYLOAD, MAX_TCP_CLIENTS};

    #[test]
    fn oversized_payload_length_is_rejected_before_allocation() {
        let mut header = [0u8; 14];
        header[10..14].copy_from_slice(&((MAX_FRAME_PAYLOAD as u32) + 1).to_le_bytes());
        let err = payload_len(&header).unwrap_err();
        assert!(err.to_string().contains("exceeds"));
        header[10..14].copy_from_slice(&(MAX_FRAME_PAYLOAD as u32).to_le_bytes());
        assert_eq!(payload_len(&header).unwrap(), MAX_FRAME_PAYLOAD);
    }

    #[test]
    fn tcp_client_limit_stops_extra_connections() {
        let active = AtomicUsize::new(0);
        for _ in 0..MAX_TCP_CLIENTS {
            assert!(try_admit_tcp_client(&active));
        }
        assert!(!try_admit_tcp_client(&active));
        release_tcp_client(&active);
        assert!(try_admit_tcp_client(&active));
    }
}
