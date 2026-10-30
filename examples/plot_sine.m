t = linspace(0, 2 * pi, 64);
y = sin(t);
figure(1);
plot(t, y);
hold('on');
stem(t, y);
xlabel('t');
ylabel('sin');
title('sine');
