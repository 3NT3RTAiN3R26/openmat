# Changelog

Versioning follows SemVer. Each toolbox port is a minor release, fixes are patch releases.

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
