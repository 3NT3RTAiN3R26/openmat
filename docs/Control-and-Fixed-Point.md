# Control and fixed-point

## Control

`tf`, `ss`, `step`, `bode`, `pole`. `step` integrates controllable canonical form with RK4.

## Fixed-point

`fi(value, signed, word length, fraction length)`, `fi2hex`, `fi_write(x, 'file.txt')`, `double`.

Rounding is half away from zero. Overflow saturates. `fi_write` writes one hex word per line, column-major, for VHDL testbenches.
