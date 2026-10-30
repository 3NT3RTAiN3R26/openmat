use std::io::{Read, Write};
use std::net::{TcpStream, UdpSocket};
use std::time::Duration;

use crate::device::{ensure_ack, Device};
use crate::protocol::{decode, encode, Frame};
use crate::HilError;

pub fn exchange_udp(socket: &UdpSocket, frame: &Frame) -> Result<Frame, HilError> {
    let bytes = encode(frame);
    socket
        .send(&bytes)
        .map_err(|err| HilError::new(format!("UDP send failed: {err}")))?;
    let mut buf = [0u8; 65535];
    let n = socket
        .recv(&mut buf)
        .map_err(|err| HilError::new(format!("UDP recv failed: {err}")))?;
    ensure_ack(decode(&buf[..n])?)
}

pub fn exchange_tcp(stream: &mut TcpStream, frame: &Frame) -> Result<Frame, HilError> {
    let bytes = encode(frame);
    stream
        .write_all(&bytes)
        .map_err(|err| HilError::new(format!("TCP write failed: {err}")))?;
    read_frame(stream)
}

pub fn read_frame(stream: &mut TcpStream) -> Result<Frame, HilError> {
    let mut header = [0u8; 14];
    stream
        .read_exact(&mut header)
        .map_err(|err| HilError::new(format!("TCP header read failed: {err}")))?;
    if &header[0..4] != crate::protocol::MAGIC {
        return Err(HilError::new("TCP HIL frame has a bad magic number"));
    }
    let len = u32::from_le_bytes(header[10..14].try_into().unwrap()) as usize;
    let mut rest = vec![0u8; len + 4];
    stream
        .read_exact(&mut rest)
        .map_err(|err| HilError::new(format!("TCP body read failed: {err}")))?;
    let mut all = header.to_vec();
    all.extend_from_slice(&rest);
    ensure_ack(decode(&all)?)
}

pub fn udp_socket(host: &str, port: u16) -> Result<UdpSocket, HilError> {
    let socket = UdpSocket::bind("0.0.0.0:0")
        .map_err(|err| HilError::new(format!("cannot bind UDP socket: {err}")))?;
    socket
        .set_read_timeout(Some(Duration::from_secs(2)))
        .ok();
    socket
        .set_write_timeout(Some(Duration::from_secs(2)))
        .ok();
    socket
        .connect(format!("{host}:{port}"))
        .map_err(|err| HilError::new(format!("cannot connect UDP to {host}:{port}: {err}")))?;
    Ok(socket)
}

pub fn tcp_stream(host: &str, port: u16) -> Result<TcpStream, HilError> {
    let stream = TcpStream::connect(format!("{host}:{port}"))
        .map_err(|err| HilError::new(format!("cannot connect TCP to {host}:{port}: {err}")))?;
    stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(2))).ok();
    Ok(stream)
}

pub fn apply_locally(device: &mut Device, frame: Frame) -> Result<Frame, HilError> {
    ensure_ack(device.dispatch(frame))
}
