clear;
home;

x = [1 0 0 0];
X = fft(x);
y = ifft(X);
z = filter([1], [1], [1 2 3 4]);
c = conv([1 2], [3 4]);
[b, a] = butter(1, 0.5);
h = fir1(8, 0.2);
sys = tf([1], [1 1]);
[t, step_y] = step(sys);
p = pole(sys);
bode(sys);
