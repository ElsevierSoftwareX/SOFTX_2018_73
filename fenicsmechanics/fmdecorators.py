# Wrapper to open and close an HDF5 File.
def hdf5_file(mode="w"):
    def wrapper_func(func):
        def wrapped_func(hdf=None, *args, **kwargs):
            import dolfin as dlf
            from fenicsmechanics.dolfincompat import MPI_COMM_WORLD
            if isinstance(hdf, str):
                if mode == "a":
                    try:
                        f = dlf.HDF5File(MPI_COMM_WORLD, hdf, mode)
                    except RuntimeError:
                        f = dlf.HDF5File(MPI_COMM_WORLD, hdf, "w")
                else:
                    f = dlf.HDF5File(MPI_COMM_WORLD, hdf, mode)
            elif isinstance(hdf, dlf.HDF5File):
                f = hdf
            else:
                raise TypeError("What file are you trying to use?")
            ret = func(f, *args, **kwargs)
            f.close()
            return ret
        return wrapped_func
    return wrapper_func
