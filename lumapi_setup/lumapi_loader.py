"""
lumapi_loader.py

Robustly imports Lumerical's `lumapi` module for Ansys Lumerical 2024 R1 (v241)
on Windows, without requiring you to permanently edit PYTHONPATH or copy files
around. Import this *before* anything else that touches lumapi.

Usage:
    from lumapi_setup.lumapi_loader import lumapi
    fdtd = lumapi.FDTD(hide=False)

Why this exists:
    lumapi.py lives inside the Lumerical install tree, not in site-packages,
    so a plain `import lumapi` fails unless you've added it to sys.path.
    This loader finds it once, validates the path, and imports it via
    importlib so it behaves like a normal module afterward.
"""

import importlib.util
import os
import sys

# Default install path for v241 on Windows. Edit this if you installed
# Lumerical somewhere non-default.
DEFAULT_LUMAPI_PATH = r"C:\Program Files\Lumerical\v241\api\python\lumapi.py"


def _find_lumapi_path():
    """Look in the default v241 location first, then fall back to scanning
    C:\\Program Files\\Lumerical\\v2** for any lumapi.py, in case the user
    has multiple versions installed."""
    if os.path.exists(DEFAULT_LUMAPI_PATH):
        return DEFAULT_LUMAPI_PATH

    base = r"C:\Program Files\Lumerical"
    if os.path.isdir(base):
        for entry in sorted(os.listdir(base), reverse=True):  # prefer newest version string
            candidate = os.path.join(base, entry, "api", "python", "lumapi.py")
            if os.path.exists(candidate):
                return candidate

    raise FileNotFoundError(
        "Could not locate lumapi.py. Checked default path:\n"
        f"  {DEFAULT_LUMAPI_PATH}\n"
        "If Lumerical is installed somewhere else, edit DEFAULT_LUMAPI_PATH "
        "in lumapi_setup/lumapi_loader.py to point at your "
        "<install_dir>\\api\\python\\lumapi.py"
    )


def _load_lumapi():
    path = _find_lumapi_path()
    spec = importlib.util.spec_from_file_location("lumapi", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lumapi"] = module  # so subsequent `import lumapi` elsewhere reuses this
    spec.loader.exec_module(module)
    return module


lumapi = _load_lumapi()


if __name__ == "__main__":
    # Quick smoke test: opens an FDTD session and closes it immediately.
    # Run this once after setup to confirm lumapi + your license are working
    # before running any real simulation.
    print(f"lumapi loaded from: {lumapi.__file__}")
    print("Opening a test FDTD session (this will consume a GUI license briefly)...")
    fdtd = lumapi.FDTD(hide=True)
    print("FDTD session opened successfully.")
    fdtd.close()
    print("Session closed. lumapi is working correctly.")