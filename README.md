# openmat

**Notice:** MATLAB is a trademark of MathWorks. openmat is not affiliated with MathWorks.

openmat is an open-source interpreter for a MATLAB-like language, written in Rust. It runs `.m` scripts: matrices, plots, signal processing, control systems, fixed-point, and a hardware-in-the-loop interface for FPGAs.

License: [Apache License 2.0](LICENSE). Copyright 2026 Tobias Zoske.

This is not a complete MATLAB replacement. Simulink and the proprietary toolboxes are missing. The supported subset is listed below.

## Build and run

`rust-toolchain.toml` selects Rust 1.99. With rustup, `cargo build` is enough. Without rustup you need Rust 1.85 or newer:

```bash
cargo build --workspace
cargo test --workspace
cargo run -p openmat-cli -- examples/linalg.m
cargo run -p openmat-cli -- -e "eig([2 0; 0 3])"
cargo run -p openmat-gui
cargo run -p openmat-gui -- examples/plot_sine.m
cargo run -p openmat-agent -- --proto udp --bind 0.0.0.0:5000
```

Without a file, `openmat` starts a REPL when the terminal is interactive. Otherwise it reads stdin. A path after `openmat-gui` loads that file into the editor and shows its folder on the left.

## GUI

The desktop window has the file browser on the left, the editor, command window, and status in the center, and the workspace on the right. Figures with content open in their own windows.

The command window shows a `>>` prompt. Enter runs that line in the same session as **Run**, so variables and figures stay. An open `if`, `for`, `while`, `switch`, `function`, bracket, or a line ending in `...` waits for more input (`..`). Errors are printed and the session continues. Arrow up and arrow down walk through previous commands. `clear`, `home`, and `close` affect this session. The prompt is disabled while the debugger is running.

**Files** lists the current folder with Name, Size, Date Modified, and Type. Click a column header to sort. **Up** or a double-click on `..` goes up one level. Double-click a folder to open it. Double-click a `.m` file to load it into the editor. **Open…** and Ctrl+O open the file dialog. **Save** and Ctrl+S write the file. If there is no path yet, a dialog asks for the name. A file can also be dropped onto the window.

The editor shows line numbers. Click a number to set or clear a breakpoint (red dot). Ctrl+I indents `if`, `for`, `while`, `switch`, and `function`, and puts several commands separated by `;` on their own lines. Semicolons inside `[]` stay put, so `[1, 2; 3, 4]` remains one matrix. **Lint** under the editor reports syntax errors, more than one command on a line, tabs, trailing whitespace, and `break` / `continue` outside a loop. A missing `;` is not an error. The line runs and its result is printed.

**Run** executes the script and stops at breakpoints. **Step** starts at the first statement and then advances one statement, including inside loops and functions. The current line is highlighted and the workspace shows the variables computed so far. **Stop** aborts and keeps the variables already set. The editor is locked during a session.

**Workspace** shows Name, Value, Size, and Class after a run. Scalars appear as numbers. Small matrices appear in the Value column as a MATLAB literal: elements separated by `,`, rows by `;`, for example `[2,4,8]` and `[2,4,8;4,8,16;6,12,24]`. Larger matrices stay blank until you double-click them. A double-click opens the variable as a table, including `fi`, cells, and structs. Constants such as `pi` and `i` are not listed.

## Language

- Expressions, assignments, `ans`, output suppression with `;`
- Matrix literals: `[1 -2]` is two elements, `[1 - 2]` and `[1-2]` are subtraction
- 1-based indexing, `end`, `:`
- `if` / `elseif` / `else`, `for`, `while`, `switch` / `case` / `otherwise`
- `function` files and anonymous functions `@(x) x+1`
- Comments `%` and `%{ %}`, line continuation `...`
- Values: real and complex matrices (column-major), strings, logicals, cells, structs, function handles, `fi`, `tf` / `ss`

## Functions

Linear algebra and basics: `zeros`, `ones`, `eye`, `size`, `length`, `numel`, `ndims`, `isempty`, `diag`, `linspace`, `reshape`, `sum`, `mean`, `min`, `max`, `inv`, `det`, `eig`, `svd`, `\`, `/`, `rand`, `randn`, `rng`, `abs`, `sqrt`, `sin`, `cos`, `tan`, `exp`, `log`, `log10`, `floor`, `ceil`, `round`, `sign`, `real`, `imag`, `conj`, `transpose`, `ctranspose`, `disp`, `fprintf`, `sprintf`, `clear`, `home`.

`clear` removes variables from the workspace. `clear name` removes one name. `clear` and `clear all` empty the workspace. `pi`, `i`, `j`, `inf`, `nan`, `true`, and `false` stay. `home` clears the console output so far; text after `home` remains. Both work without parentheses: `clear x` and `home`.

Plots: `plot`, `stem`, `figure`, `clf`, `close`, `hold`, `xlabel`, `ylabel`, `title`, `subplot`. `clf` and `clf reset` clear the current figure. `close`, `close(n)`, and `close all` close figures. The GUI opens a separate window for every figure that has content. You can drag that window onto the desktop. The close button does the same as `close(n)`. The project default is `docked`: `set(0, 'DefaultFigureWindowStyle', 'docked')`. Figures created after that appear as tabs in one shared window. `set(0, 'DefaultFigureWindowStyle', 'normal')` opens them as separate windows. This applies to figures created after the `set`. A figure that already has curves keeps its current presentation. Under **Options**, **Reuse Figure** draws into the current figure. **New Figure** opens a new figure before `plot`, `stem`, `step`, and `bode` when the current figure already has curves and `hold` is off. `figure(n)` selects that figure in both modes. The CLI keeps the data and does not abort.

Signal processing: `fft`, `ifft`, `filter`, `conv`, `freqz`, `butter`, `fir1`. `butter` uses the bilinear transform with prewarping. `wn = 1` is the Nyquist frequency. Call: `[b, a] = butter(n, wn)`.

Control: `tf`, `ss`, `step`, `bode`, `pole`. `step` integrates controllable canonical form with RK4.

Fixed-point: `fi(value, signed, word length, fraction length)`, `fi2hex`, `fi_write(x, 'file.txt')`, `double`. Rounding is half away from zero. Overflow saturates. `fi_write` writes one hex word per line, column-major, for VHDL testbenches.

## HIL

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

### PYNQ RFSoC 4x2

1. Build `openmat-agent` for `aarch64-unknown-linux-gnu` and copy it to the board. On Windows, Zig is the linker, because `aarch64-unknown-linux-gnu` expects a Linux `cc`.
2. On the board: `./openmat-agent --proto udp --bind 0.0.0.0:5000`. The message `listening on 0.0.0.0:5000 (udp)` means the agent is ready.
3. `hil.open('udp', host, port)` takes the host and port from the call, not from `ethernet.host` in the board profile. `examples/hil.m` talks to `192.168.178.98` port 5000. The address in the script is the board's address. `fprintf` prints `v` and `y`.
4. The supplied agent keeps registers and samples in memory. The VHDL sources under `hdl/` (`openmat_axi_regs`, `openmat_fifo`, `openmat_packet`) are the PL counterpart: AXI register bank, FIFO, and frame parser. Addresses 0, 4, 8, and 12 are scratch, length, status, and control.

JTAG uses Vivado in batch mode and JTAG-to-AXI. `OPENMAT_VIVADO` points at the `vivado` executable. Without Vivado, `hil.open('jtag', ...)` fails with a clear message. `hil.open('jtag-mock', 'pynq_rfsoc4x2')` checks the profile without hardware. Sample streams use Ethernet. Register access uses Ethernet or JTAG.

The loopback transport and the protocol tests run without a board: `cargo run -p openmat-cli -- examples/hil_loopback.m`. Against the board: `cargo run -p openmat-cli -- examples/hil.m`, while the agent listens on UDP port 5000.

## Examples

Every script starts with `clear` and `home`.

- `examples/linalg.m`
- `examples/plot_sine.m` draws sine and cosine in two docked figures
- `examples/dsp_demo.m`
- `examples/fixed_point.m`
- `examples/matrix_demo.m` shows `clear`, `home`, and `a .* b` for a row and a column
- `examples/hil_loopback.m`
- `examples/hil.m`
