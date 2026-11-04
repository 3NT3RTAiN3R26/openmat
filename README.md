# openmat

**Notice:** MATLAB is a trademark of MathWorks. openmat is not affiliated with MathWorks.

openmat is an open-source interpreter for a MATLAB-like language, written in Rust. It runs `.m` scripts: matrices, plots, signal processing, control systems, fixed-point, and a hardware-in-the-loop interface for FPGAs.

License: [Apache License 2.0](LICENSE). Copyright 2026 Tobias Zoske.

This is not a complete MATLAB replacement. Simulink is missing. The built-in subset is in the [wiki](https://github.com/3NT3RTAiN3R26/openmat/wiki).

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

## Supported areas

The function lists and calling details are in the [wiki](https://github.com/3NT3RTAiN3R26/openmat/wiki).

- [Language](https://github.com/3NT3RTAiN3R26/openmat/wiki/Language): scripts, matrices, cells, structs, `fi`, `tf` / `ss`
- [Core](https://github.com/3NT3RTAiN3R26/openmat/wiki/Core): arrays, linear algebra, text, plots
- [Signal Processing](https://github.com/3NT3RTAiN3R26/openmat/wiki/Signal-Processing): windows, filtering, IIR and FIR design, spectra, resampling. Not yet: `cwt`, `dwt`
- [Control and fixed-point](https://github.com/3NT3RTAiN3R26/openmat/wiki/Control-and-Fixed-Point): `tf`, `ss`, `step`, `bode`, `fi`
- [Search path](https://github.com/3NT3RTAiN3R26/openmat/wiki/Search-Path): `addpath`, `private/`, `+package`. Not yet: `@class`
- [HIL](https://github.com/3NT3RTAiN3R26/openmat/wiki/HIL): loopback, UDP, TCP, and JTAG against a board profile

## Search path

A name that is not a variable or a function in the script is looked up as an `.m` file in the current folder, then on the search path. `addpath`, `rmpath`, `path`, `genpath`, `which`, `exist`, `run`, and `savepath` are built in. The CLI adds the script folder to the path. `openmat --cd script.m` changes into that folder instead. Details and `private/` / `+package` rules are on the [search path](https://github.com/3NT3RTAiN3R26/openmat/wiki/Search-Path) page.

## HIL

`hil.open`, `hil.write`, `hil.read`, `hil.regwrite`, `hil.regread`, and `hil.close` talk to a loopback target or to `openmat-agent` over UDP or TCP. JTAG uses Vivado. Board profiles live in `boards/`. The frame layout and the PYNQ RFSoC 4x2 steps are on the [HIL](https://github.com/3NT3RTAiN3R26/openmat/wiki/HIL) page. Without a board: `cargo run -p openmat-cli -- examples/hil_loopback.m`.

## Examples

Every script starts with `clear` and `home`.

- `examples/linalg.m`
- `examples/plot_sine.m` draws sine and cosine in two docked figures
- `examples/dsp_demo.m`
- `examples/fixed_point.m`
- `examples/matrix_demo.m` shows `clear`, `home`, and `a .* b` for a row and a column
- `examples/hil_loopback.m`
- `examples/hil.m`
- `examples/path_demo/main.m` calls a function from `examples/path_demo/lib` after `addpath`
