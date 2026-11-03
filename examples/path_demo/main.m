clear
home
% Search path demo: functions in lib/ are found after addpath.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'lib'));
fprintf('%s\n', which('scale_by'));
disp(scale_by([1 2 3], 10));
rmpath(fullfile(here, 'lib'));
