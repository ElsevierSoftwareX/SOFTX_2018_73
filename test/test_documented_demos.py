import os
import pytest
import fnmatch
import subprocess as sub

import fenicsmechanics as fm
fm_dirname, _ = os.path.split(fm.__file__)
base_demo_dir = os.path.abspath(os.path.join(fm_dirname, "../demos"))

# SKIPPING ELLIPSOID DUE TO RESOURCES ON LOCAL MACHINE.
demos_to_test = (# "ellipsoid",
    "pipe_flow",
    "unloading",
    "square_elongation")

@pytest.mark.parametrize("demo", demos_to_test)
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

if __name__ == "__main__":
    import sys
    import argparse

    try: # For use with emacs python shell
        sys.argv.remove("--simple-prompt")
    except ValueError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-ellipsoid",
                        action="store_true",
                        help="Include the ellipsoid demo.")
    args = parser.parse_args()

    timeout = 100
    if args.test_ellipsoid:
        demos_to_test = ("ellipsoid",) + demos_to_test
        timeout = 5000 # incresing timeout for ellipsoid

    for demo in demos_to_test:
        test_demo(demo, timeout=timeout)
