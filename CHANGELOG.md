# Changelog

Versioning follows SemVer. Each toolbox port is a minor release, fixes are patch releases.

## 0.3.0

- Windows: `rectwin`, `bartlett`, `hann`, `hamming`, `blackman`, `kaiser`, `gausswin`, `tukeywin`.
- Basics: `nextpow2`, `fftshift`, `ifftshift`, `sinc`, `angle`, `cumsum`, `diff`, `unwrap`, `detrend`, `hilbert`, `xcorr`.
- Polynomials: `polyval`, `poly`, `roots`, `deconv`.
- Filtering: `filter` with initial conditions, final state, dimension, and the shape of the input (it returned a column before), `filtic`, `filtfilt`, `sosfilt`.
- Conversions: `sos2tf`, `zp2tf` (pads the numerator like MATLAB), `tf2zp` (error for an improper system). Responses: `freqs`, `freqz` (MATLAB grid, `fs`, `whole`, point vectors), `phasez`, `grpdelay`, `impz`, `stepz` with the lengths MATLAB R2025a returns. `roots` lists the positive imaginary part of a conjugate pair first. `conv` and `deconv` follow the orientation of the first input.
- `conv` accepts `'full'`, `'same'`, `'valid'`. New: `cov`, `corrcoef`, `xcov`.
- IIR design: `butter` (low, high, bandpass, stop, analog, `[z, p, k]`, and `[A, B, C, D]`), `cheby1`, `cheby2`, `ellip`, `besself`, `buttord`, `cheb1ord`, `cheb2ord`, `ellipord`, `bilinear`, `iirnotch`, `iirpeak`, `iircomb` (notch or peak, with `Ab`). `butter` with one output returns `b` instead of an error. Analog numerators are padded to the length of the denominator, and `[z, p, k]` come in MATLAB's order. Four outputs realize the unit-cutoff prototype in state space, apply `lp2lp` / `lp2hp` / `lp2bp` / `lp2bs`, then the bilinear transform at `fs = 2`. `buttord` returns the stopband-matched 3 dB edge; `cheb1ord` and `ellipord` return `Wp`; `cheb2ord` returns `Ws`. `bilinear` follows MATLAB: two outputs give the transfer function form, three the zpk form, with an optional prewarp frequency. `iirnotch` and `iirpeak` accept `Ab`.
- FIR design: `fir1` and `fir2` (numeric `npt`, `lap`, window, and the odd-order Nyquist warning; a trailing string is rejected as a non-numeric `npt`), `firls` (with hilbert, differentiator and weights), `firpm` / `firpmord`, `kaiserord` (`low`, `high`, `DC-0`, `stop`, and `beta = 0` at 20 dB), `firceqrip` (transition centered on `Fo`; the `'slope'` tilt is approximate), `sgolay` (weights and the `G` output), `sgolayfilt` (weights), `gaussdesign`, `rcosdesign`. The even-order differentiator from `firpm` matches MATLAB to about `5e-4`.
- Filter analysis: `tf2sos` and `zp2sos` (`up` / `down`, gain `g`, and `inf` / `two` scaling). The scale is the cumulative section norm: `g = 1/||1/A1||` and each numerator is the ratio of successive norms, so the posted `b1` infinity-norm and 2-norm gains match. Repeated zeros move the numerator coefficients by about `1e-3` because the companion-matrix roots are not exact. `residuez` covers repeated poles, the direct term, and the inverse call. `zplane` draws the unit circle and pole/zero markers when there is no output. There is no graphics-handle type, so outputs of `zplane` are empty.
- Spectral analysis: `periodogram`, `pwelch`, `cpsd`, `mscohere`, `tfestimate` (window, overlap, `nfft`, `fs`, `onesided`/`twosided`, `psd`/`power`), `spectrogram` (`[S, F, T, P]`), `bandpower`, `enbw`, `snr`, `thd`, `sfdr`, `sinad` (Kaiser window with beta 38, harmonics up to the sixth). `stft` and `istft` (Window, OverlapLength, FFTLength, FrequencyRange, ConjugateSymmetric), `pspectrum` (power spectrum with Leakage and FrequencyLimits, and `'spectrogram'` with the default 1024-by-time layout). With no output arguments, `periodogram`, `pwelch`, `cpsd`, `mscohere`, `tfestimate`, `spectrogram`, `pspectrum`, `freqz`, `phasez` and `grpdelay` draw a magnitude, phase or delay curve. The figure model is line-only, so a spectrogram is that curve (the mean over time) rather than a heatmap. `FrequencyResolution` is not implemented. Not yet: `cwt`, `dwt`.
- Resampling: `upsample`, `downsample`, `upfirdn`, `interp` (band-limited filter with odd edge extension, returns the filter as a second output), `decimate` (Chebyshev IIR with zero-phase filtering or `'fir'`, the kept samples end on the last sample; an IIR order above 13 warns, the same as MATLAB, which does not stage large factors itself), `resample` (Kaiser-windowed FIR, `[y, b]`).
- Other: `findpeaks` (`MinPeakHeight`, `MinPeakDistance`, `Threshold`, `NPeaks`, `SortStr`, `MinPeakProminence`, `MinPeakWidth`, `MaxPeakWidth`, sample rate or locations, `[pks, locs, w, p]`), `medfilt1` (zero padding or truncation, `omitnan`), `envelope` (`'analytic'` with the Hilbert transform or an N-tap FIR, `'rms'`, `'peak'` / `'peaks'`), `fft2`, `ifft2`, `fft(x, n, dim)` and `ifft(x, n, dim)`. `ifft` returns a real result when the input is conjugate symmetric and no longer rounds small values to zero. Not yet: `cwt`, `dwt`.

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
