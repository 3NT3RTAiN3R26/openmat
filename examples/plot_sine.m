clear;
home;

set(0, 'DefaultFigureWindowStyle', 'docked')


t = linspace(0, 2 * pi, 64);
y = sin(t);
figure(1);
plot(t, y);
hold('on');
stem(t, y);
xlabel('t');
ylabel('sin');
title('sine');

z = cos(t);
figure(2);
plot(t, z);
hold('on');
stem(t, z);
xlabel('t');
ylabel('cos');
title('cosine');


