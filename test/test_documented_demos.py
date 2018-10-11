import os
import pytest
import fnmatch
import subprocess as sub

import fenicsmechanics as fm
fm_dirname, _ = os.path.split(fm.__file__)
base_demo_dir = os.path.abspath(os.path.join(fm_dirname, "../demos"))

# SKIPPING ELLIPSOID DUE TO RESOURCES ON LOCAL MACHINE.
@pytest.mark.parametrize("demo", (# "ellipsoid",
                                  "pipe_flow",
                                  "unloading",
                                  "square_elongation"))
def test_demo(demo, timeout=200):
    py_cmd = "python3 {fname}"
    demo_dir = os.path.join(base_demo_dir, demo)
    os.chdir(demo_dir)
    cwd = os.getcwd()
    for fname in os.listdir(cwd):
        returncode = 0
        if fnmatch.fnmatch(fname, "*.py"):
            cmd = py_cmd.format(fname=fname).split()
            ret = sub.run(cmd, timeout=timeout)
            returncode = ret.returncode
        assert not returncode
    return None
