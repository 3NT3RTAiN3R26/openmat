# Search path and packages

A name that is not a variable or a function defined in the script is looked up as an `.m` file in the current folder and then in the search path, first entry first. A function file defines the function named after the file. A script file runs in the caller's workspace. Edited files are read again on the next run.

```matlab
addpath('lib')                 % add to the front, same as '-begin'
addpath('lib', '-end')         % add to the end; 0 and 1 work like '-begin' and '-end'
old = addpath('a', 'b')        % returns the previous path
addpath(genpath('toolbox'))    % a folder and its subfolders, without private, @ and + folders
rmpath('lib'); path            % path lists, path(p) sets
which name; exist('name')      % 1 variable, 2 file, 5 built-in, 7 folder, 103 command-line function
run('script'); mfilename       % run a file in this workspace; name of the running file
savepath; restoredefaultpath   % keep the path for the next start; empty the path
```

Also available: `pwd`, `cd`, `fullfile`, `fileparts`, `pathsep`, `filesep`. Relative folders are made absolute when added. A folder that does not exist gives a warning and is skipped.

`savepath` writes `pathdef.txt` in `%APPDATA%\openmat` (`$XDG_CONFIG_HOME/openmat` or `~/.config/openmat` elsewhere). `OPENMAT_PATHDEF` points at another file. The CLI and the GUI load it at start.

The CLI adds the folder of the script to the path. `openmat --cd script.m` changes into that folder instead. In the GUI, **Run** and **Step** ask whether to **Change Folder** or **Add to Path** when the script's folder is neither the current folder nor on the path. **Path...** edits the search path.

Folders named `private` hold functions that only files in the parent folder, and other files in that `private` folder, can call. They win over functions of the same name on the path. A `+pkg` folder is a package: `+pkg/fn.m` is called as `pkg.fn(...)`, and `+a/+b/fn.m` as `a.b.fn(...)`. The extra functions in a function file are local to that file.

Not supported yet: `@class` folders.
