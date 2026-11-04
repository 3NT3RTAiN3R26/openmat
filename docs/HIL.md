# Hardware in the loop

```matlab
h = hil.open('loopback');
h = hil.open('udp', '192.168.2.99', 5000);
h = hil.open('tcp', '192.168.2.99', 5000);
h = hil.open('jtag', 'pynq_rfsoc4x2');
hil.write(h, [1 2 3]);
y = hil.read(h, 3);
hil.regwrite(h, 0, 42);
v = hil.regread(h, 0);
hil.close(h);
```

Frame: magic `OMAT`, version, opcode, address, length, payload, CRC-32. Opcodes: Write=1, Read=2, RegWrite=3, RegRead=4, Ack=5, Err=6. Samples are little-endian `f64`.

Board profiles live in `boards/`. `boards/pynq_rfsoc4x2.toml` describes the PYNQ RFSoC 4x2 (XCZU48DR). `boards/explorer.toml` is only the format for the FPGA Horizons Explorer board (spring 2027). A new board is another TOML file, not a code change. `OPENMAT_BOARDS` points at another directory.

## PYNQ RFSoC 4x2

1. Build `openmat-agent` for `aarch64-unknown-linux-gnu` and copy it to the board. On Windows, Zig is the linker, because `aarch64-unknown-linux-gnu` expects a Linux `cc`.
2. On the board: `./openmat-agent --proto udp --bind 0.0.0.0:5000`. The message `listening on 0.0.0.0:5000 (udp)` means the agent is ready.
3. `hil.open('udp', host, port)` takes the host and port from the call, not from `ethernet.host` in the board profile. `examples/hil.m` talks to `192.168.178.98` port 5000. The address in the script is the board's address. `fprintf` prints `v` and `y`.
4. The supplied agent keeps registers and samples in memory. The VHDL sources under `hdl/` (`openmat_axi_regs`, `openmat_fifo`, `openmat_packet`) are the PL counterpart: AXI register bank, FIFO, and frame parser. Addresses 0, 4, 8, and 12 are scratch, length, status, and control.

JTAG uses Vivado in batch mode and JTAG-to-AXI. `OPENMAT_VIVADO` points at the `vivado` executable. Without Vivado, `hil.open('jtag', ...)` fails with a clear message. `hil.open('jtag-mock', 'pynq_rfsoc4x2')` checks the profile without hardware. Sample streams use Ethernet. Register access uses Ethernet or JTAG.

The loopback transport and the protocol tests run without a board: `cargo run -p openmat-cli -- examples/hil_loopback.m`. Against the board: `cargo run -p openmat-cli -- examples/hil.m`, while the agent listens on UDP port 5000.
