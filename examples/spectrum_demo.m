fs = 1000;
t = (0:999) / fs;
x = sin(2*pi*50*t) + 0.1 * sin(2*pi*150*t);
[pxx, f] = pwelch(x, 250, 125, 256, fs);
peak = f(pxx == max(pxx))
power = bandpower(x, fs, [40 60])
[s, f2, t2] = spectrogram(x, 128, 64, 128, fs);
size(s)
thd_db = thd(x)
