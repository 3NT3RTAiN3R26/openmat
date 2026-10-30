h = hil.open('loopback');
hil.regwrite(h, 0, 42);
v = hil.regread(h, 0);
hil.write(h, [1 2 3 4]);
y = hil.read(h, 4);
hil.close(h);
