clear;
home;

x = fi(0.5, 1, 16, 15);
hex = fi2hex(x);
y = x + fi(0.25, 1, 16, 15);
back = double(y);
word = x.WordLength;
low = fi(1.9, 1, 8, 0, 'Floor', 'Saturate');
wrapped = ufi(-1, 8, 0, 'Wrap');
bits = bin(wrapped);
