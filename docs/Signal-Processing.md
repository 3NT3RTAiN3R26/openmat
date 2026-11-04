# Signal Processing

Compared with MATLAB R2025a. A few designs are close rather than identical. Those cases are named below.

## Windows and basics

Column vectors. `'periodic'` is accepted by `hann`, `hamming`, and `blackman`. `bartlett` does not take `'periodic'`.

`rectwin`, `bartlett`, `hann`, `hamming`, `blackman`, `kaiser(L, beta)`, `gausswin(L, alpha)`, `tukeywin(L, r)`.

`nextpow2`, `fftshift`, `ifftshift`, `sinc`, `angle`, `cumsum`, `diff`, `unwrap`, `detrend` (`'constant'`, `'linear'`, or a polynomial order), `hilbert`, `xcorr` (`[c, lags] = xcorr(x, y, maxlag, scale)` with `none`, `biased`, `unbiased`, `coeff`).

`polyval`, `poly`, `roots`, `[q, r] = deconv(y, a)`. `roots` lists a conjugate pair with the positive imaginary part first.

`fft`, `ifft`, `fft2`, `ifft2`, including `fft(x, n, dim)` and `ifft(x, n, dim)`. `ifft` returns a real result when the input is conjugate symmetric.

## Filtering

`y = filter(b, a, x)` keeps the shape of `x`. `[y, zf] = filter(b, a, x, zi, dim)` takes and returns the filter state. `zi = filtic(b, a, y_past, x_past)`.

`filtfilt`, `sosfilt`. `conv(a, b, 'full'|'same'|'valid')` follows the orientation of the first input.

Conversions: `sos2tf`, `zp2tf` (pads the numerator), `tf2zp` (errors on an improper system), `tf2sos`, `zp2sos` (`up` / `down`, gain `g`, `inf` / `two`). Repeated zeros can move `tf2sos` numerators by about `1e-3`. `residuez` covers repeated poles, the direct term, and the inverse call.

Responses: `freqz` (grid `(0:n-1)*pi/n`, 512 points by default, `fs`, `'whole'`, or a vector of points), `phasez`, `grpdelay`, `impz`, `stepz`, `freqs`. A point count for `freqs` selects a logarithmic grid. With no outputs, `freqz`, `phasez`, and `grpdelay` draw a curve. `zplane` draws the unit circle and pole/zero markers when there is no output. There is no graphics-handle type, so the outputs of `zplane` are empty.

Statistics: `cov`, `corrcoef`, `xcov`.

## IIR design

`butter`, `cheby1`, `cheby2`, `ellip`, `besself`. Types `'low'`, `'high'`, `'bandpass'`, `'stop'`, and `'s'` for analog filters. A two-element `Wn` means bandpass or stop and doubles the order. `Wn = 1` is Nyquist. One output returns `b`. Two outputs return `[b, a]`. Three return `[z, p, k]` in MATLAB's order. Four return `[A, B, C, D]`.

`buttord` returns the 3 dB edge. `cheb1ord` and `ellipord` return `Wp`. `cheb2ord` returns `Ws`.

`bilinear` uses the number of outputs to select the transfer-function form or the zpk form. An optional `fp` prewarps one frequency.

`iirnotch`, `iirpeak`, and `iircomb` take an optional bandwidth level `Ab` (3 dB by default).

`lp2lp`, `lp2hp`, `lp2bp`, `lp2bs` transform a state-space system.

`decimate` uses one Chebyshev stage. An IIR order above 13 warns. MATLAB does not stage large factors itself.

## FIR design

`fir1` and `fir2` (`'low'`, `'high'`, `'bandpass'`, `'stop'`, `'DC-0'`, `'DC-1'`, window, `'scale'` / `'noscale'`, numeric `npt`). An odd order that passes Nyquist is raised by one, with a warning. A trailing string where `fir2` expects `npt` is an error.

`firls` (weights, `'hilbert'`, `'differentiator'`). `firpm`, `firpmord`. The even-order `firpm` differentiator matches MATLAB to about `5e-4`. `kaiserord` (`low`, `high`, `DC-0`, `stop`; `beta = 0` at 20 dB). `firceqrip` centers the transition on `Fo`. The `'slope'` tilt is approximate.

`sgolay` (weights and the `G` output), `sgolayfilt`. `gaussdesign`, `rcosdesign` (`'sqrt'` or `'normal'`, unit energy).

## Spectral analysis

`periodogram`, `pwelch`, `cpsd`, `mscohere`, `tfestimate` (window, overlap, `nfft`, `fs`, `'onesided'` / `'twosided'`, `'psd'` / `'power'`).

`spectrogram` returns `[S, F, T, P]`. `stft`, `istft` (`Window`, `OverlapLength`, `FFTLength`, `FrequencyRange`, `ConjugateSymmetric`). `pspectrum` (power spectrum with `Leakage` and `FrequencyLimits`, and `'spectrogram'`). `FrequencyResolution` is not implemented.

`bandpower`, `enbw`, `snr`, `thd`, `sfdr`, `sinad` (Kaiser window, beta 38, harmonics up to the sixth).

With no output arguments, these functions draw a magnitude, phase, or delay curve. A spectrogram with no outputs is the mean magnitude over time, not a heatmap.

## Resampling and other

`upsample`, `downsample`, `upfirdn`, `interp` (returns the filter as a second output), `decimate`, `resample` (`[y, b]`).

`findpeaks` (`MinPeakHeight`, `MinPeakDistance`, `Threshold`, `NPeaks`, `SortStr`, `MinPeakProminence`, `MinPeakWidth`, `MaxPeakWidth`, sample rate or locations, `[pks, locs, w, p]`).

`medfilt1` (zero padding or truncation, `'omitnan'`). `envelope` (`'analytic'` with the Hilbert transform or an N-tap FIR, `'rms'`, `'peak'` / `'peaks'`).

Not implemented: `cwt`, `dwt`.
