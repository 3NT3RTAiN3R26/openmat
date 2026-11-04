# Core, plots, and text

## Arrays and linear algebra

`zeros`, `ones`, `eye`, `size`, `length`, `numel`, `ndims`, `isempty`, `diag`, `linspace`, `reshape`, `sum`, `mean`, `min`, `max`, `inv`, `det`, `eig`, `svd`, `\`, `/`, `rand`, `randn`, `rng`, `abs`, `sqrt`, `sin`, `cos`, `tan`, `exp`, `log`, `log10`, `floor`, `ceil`, `round`, `sign`, `real`, `imag`, `conj`, `transpose`, `ctranspose`.

## Text

`disp`, `fprintf`, `sprintf`, `clear`, `home`.

## Plots

`plot`, `stem`, `figure`, `clf`, `close`, `hold`, `xlabel`, `ylabel`, `title`, `subplot`, `set`.

`clf` and `clf reset` clear the current figure. `close`, `close(n)`, and `close all` close figures. The GUI opens a separate window for every figure that has content. The project default is `docked`: `set(0, 'DefaultFigureWindowStyle', 'docked')` puts later figures in one window. `set(0, 'DefaultFigureWindowStyle', 'normal')` opens them separately. That applies to figures created after the `set`. Under **Options**, **Reuse Figure** draws into the current figure. **New Figure** opens a new figure before `plot`, `stem`, `step`, and `bode` when the current figure already has curves and `hold` is off. `figure(n)` selects that figure in both modes. The CLI keeps the data and does not abort.
