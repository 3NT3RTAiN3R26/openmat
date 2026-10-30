clear;
home;

% Talks to openmat-agent on the PYNQ RFSoC at UDP port 5000.
% Start the agent on the board first:
%   ./openmat-agent --proto udp --bind 0.0.0.0:5000
% Registers and samples stay in the agent process. This does not write the FPGA fabric.
h = hil.open('udp', '192.168.178.98', 5000);
hil.regwrite(h, 0, 42);
v = hil.regread(h, 0);
hil.write(h, [1 2 3 4]);
y = hil.read(h, 4);
fprintf('v = %d\n', v);
fprintf('%s\n', y);
hil.close(h);
