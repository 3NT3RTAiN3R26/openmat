# Changelog

Versioning follows SemVer. Each toolbox port is a minor release, fixes are patch releases.

## Unreleased (0.3.0, Signal Processing Toolbox)

- Windows: `rectwin`, `bartlett`, `hann`, `hamming`, `blackman`, `kaiser`, `gausswin`, `tukeywin`.
- Basics: `nextpow2`, `fftshift`, `ifftshift`, `sinc`, `angle`, `cumsum`, `diff`, `unwrap`, `detrend`, `hilbert`, `xcorr`.
- Polynomials: `polyval`, `poly`, `roots`, `deconv`.
- Filtering: `filter` with initial conditions, final state, dimension, and the shape of the input (it returned a column before), `filtic`, `filtfilt`, `sosfilt`.
- Conversions: `sos2tf`, `zp2tf` (pads the numerator like MATLAB), `tf2zp` (error for an improper system). Responses: `freqs`, `freqz` (MATLAB grid, `fs`, `whole`, point vectors), `phasez`, `grpdelay`, `impz`, `stepz` with the lengths MATLAB R2025a returns. `roots` lists the positive imaginary part of a conjugate pair first. `conv` and `deconv` follow the orientation of the first input.
- `conv` accepts `'full'`, `'same'`, `'valid'`. New: `cov`, `corrcoef`, `xcov`.
- IIR design: `butter` (low, high, bandpass, stop, analog, `[z, p, k]`), `cheby1`, `cheby2`, `bilinear`, `iirnotch`, `iirpeak`. `butter` with one output returns `b` instead of an error. Analog numerators are padded to the length of the denominator, and `[z, p, k]` come in MATLAB's order. `bilinear` follows MATLAB: two outputs give the transfer function form, three the zpk form, with an optional prewarp frequency. `iirnotch` and `iirpeak` accept `Ab`.

## 0.2.1

- `private/` folders: functions in them are visible only to files in the parent folder and to each other, and win over path functions of the same name.
- `+package` folders: `pkg.fn(...)`, nested `a.b.fn(...)`, and `pkg.fn` without parentheses.
- Local functions of a function file are private to that file. Two files can now define a local function with the same name.

## 0.2.0

- Search path: `.m` files in the current folder and on the path are found as functions and scripts.
- New functions: `addpath`, `rmpath`, `path`, `genpath`, `restoredefaultpath`, `savepath`, `pwd`, `cd`, `pathsep`, `filesep`, `which`, `exist`, `mfilename`, `run`, `fullfile`, `fileparts`.
- CLI adds the script's folder to the path, `--cd` changes into it instead. The saved path is loaded at start.
- GUI asks "Change Folder / Add to Path / Cancel" before running a script outside the path. **Path...** edits and saves the search path.
- A bare function name such as `p = path` now calls the function instead of returning a handle.
- The agent binds to `127.0.0.1:5000` by default. Use `--bind 0.0.0.0:5000` on the board.
- HIL server: oversized UDP responses become error frames, a poisoned lock no longer panics, and the UDP loop survives receive errors.

## 0.1.0

Baseline: parser and interpreter, plots and GUI, DSP and control basics, `fi`, HIL over UDP, TCP, JTAG, and loopback, VHDL sources.
