use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use clap::Parser;
use openmat_hil::{spawn_tcp, spawn_udp, Device};

/// Serves the openmat HIL protocol on the board, or on a PC with the in-memory device.
#[derive(Parser)]
#[command(name = "openmat-agent", version)]
struct Cli {
    /// udp or tcp
    #[arg(long, default_value = "udp")]
    proto: String,
    /// Bind address
    #[arg(long, default_value = "0.0.0.0:5000")]
    bind: String,
}

fn main() {
    let cli = Cli::parse();
    let device = Arc::new(Mutex::new(Device::default()));
    let addr = match cli.proto.as_str() {
        "tcp" => spawn_tcp(&cli.bind, Arc::clone(&device)).expect("tcp bind"),
        "udp" => spawn_udp(&cli.bind, Arc::clone(&device)).expect("udp bind"),
        other => {
            eprintln!("unknown protocol {other}; use udp or tcp");
            std::process::exit(2);
        }
    };
    println!("openmat-agent listening on {addr} ({})", cli.proto);
    println!("register and sample traffic stays in the in-memory device unless you extend it with /dev/mem");
    loop {
        thread::sleep(Duration::from_secs(3600));
    }
}
