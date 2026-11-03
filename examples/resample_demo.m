fs = 1000;
t = (0:999)' / fs;
x = sin(2*pi*50*t) + 0.3 * sin(2*pi*210*t);
y = decimate(x, 4);
numel(y)
z = resample(x, 3, 2);
numel(z)
u = interp(x(1:100), 2);
numel(u)
[pks, locs] = findpeaks(x, 'MinPeakHeight', 1);
numel(pks)
m = medfilt1(x, 5);
e = envelope(x);
X = fft2([16 2 3 13; 5 11 10 8; 9 7 6 12; 4 14 15 1]);
abs(X(1, 1))
