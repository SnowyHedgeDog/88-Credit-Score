"""macOS CLI startup fallback for an existing Conda OpenMP installation."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile


def bootstrap_openmp():
    """Set dyld's search path before native imports, restarting at most once.

    SnapML references Homebrew's absolute libomp path, while XGBoost uses
    @rpath. Preloading with ctypes alone fixes only XGBoost. dyld must see the
    fallback directory when Python starts to resolve both libraries.
    """
    if sys.platform != "darwin" or os.environ.get("AML_OPENMP_BOOTSTRAPPED"):
        return
    standard = [Path("/usr/local/opt/libomp/lib/libomp.dylib"),
                Path("/opt/homebrew/opt/libomp/lib/libomp.dylib")]
    search = os.environ.get("DYLD_LIBRARY_PATH", "")
    if any(path.is_file() for path in standard) or any(
        (Path(directory) / "libomp.dylib").is_file()
        for directory in search.split(os.pathsep) if directory
    ):
        return
    prefixes = [Path(sys.base_prefix)]
    if os.environ.get("CONDA_PREFIX"):
        prefixes.insert(0, Path(os.environ["CONDA_PREFIX"]))
    prefixes.extend(Path.home() / name for name in
                    ("miniconda3", "anaconda3", "miniforge3", "mambaforge"))
    library = next((prefix / "lib" / "libomp.dylib" for prefix in prefixes
                    if (prefix / "lib" / "libomp.dylib").is_file()), None)
    if library is None:
        return  # Preserve the native library's actionable installation error.

    # Expose only OpenMP: adding all of Conda's lib directory can shadow
    # unrelated NumPy/system libraries. Keep generated files inside the venv.
    directory = Path(sys.prefix) / "lib" / "aml-openmp"
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=".libomp-")
    os.close(fd)
    try:
        shutil.copyfile(library, temporary)
        os.replace(temporary, directory / "libomp.dylib")
    finally:
        Path(temporary).unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["DYLD_LIBRARY_PATH"] = str(directory) + (os.pathsep + search if search else "")
    environment["AML_OPENMP_BOOTSTRAPPED"] = "1"
    print(f"Using existing OpenMP runtime: {library}", flush=True)
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)
