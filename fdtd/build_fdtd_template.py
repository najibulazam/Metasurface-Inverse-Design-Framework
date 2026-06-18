"""
build_fdtd_template.py

Build and save a reusable Lumerical `.fsp` file for one metasurface
unit-cell configuration. This is useful for:

    1. visually inspecting the scene in the Lumerical GUI,
    2. confirming material names / monitor placement before a long sweep,
    3. creating a device or reference template that matches the scripted
       pipeline in `fdtd/unit_cell_sim.py`.
"""

import argparse
import os

from lumapi_setup.lumapi_loader import lumapi
from fdtd.unit_cell_sim import build_unit_cell

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "fdtd_templates")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Build and save a single FDTD template .fsp file.")
    p.add_argument("--x_size", type=float, default=120.0, help="pillar width along x in nm")
    p.add_argument("--y_size", type=float, default=80.0, help="pillar width along y in nm")
    p.add_argument("--height", type=float, default=450.0, help="pillar height in nm")
    p.add_argument("--start_wave", type=float, default=400.0, help="start wavelength in nm")
    p.add_argument("--end_wave", type=float, default=800.0, help="stop wavelength in nm")
    p.add_argument("--points", type=int, default=21, help="frequency points for monitors")
    p.add_argument("--mesh_accuracy", type=int, default=4, help="FDTD mesh accuracy")
    p.add_argument("--polarization_deg", type=float, default=0.0, help="source polarization angle")
    p.add_argument("--reference", action="store_true", help="omit the pillar and save a reference cell")
    p.add_argument("--hide", action="store_true", help="hide the GUI while generating the file")
    p.add_argument(
        "--output",
        default=None,
        help="absolute or relative .fsp output path; defaults under outputs/fdtd_templates/",
    )
    return p.parse_args()


def default_output_path(args):
    suffix = "reference" if args.reference else "device"
    filename = (
        f"unitcell_{suffix}_x{int(args.x_size)}_y{int(args.y_size)}"
        f"_h{int(args.height)}_pol{int(args.polarization_deg)}.fsp"
    )
    return os.path.join(OUTPUT_DIR, filename)


def main():
    args = parse_args()
    output_path = os.path.abspath(args.output or default_output_path(args))

    fdtd = lumapi.FDTD(hide=args.hide)
    try:
        build_unit_cell(
            fdtd,
            x_size_nm=args.x_size,
            y_size_nm=args.y_size,
            height_nm=args.height,
            lambda_start_nm=args.start_wave,
            lambda_stop_nm=args.end_wave,
            freq_points=args.points,
            mesh_accuracy=args.mesh_accuracy,
            source_polarization_deg=args.polarization_deg,
            include_pillar=not args.reference,
        )
        fdtd.save(output_path)
        print(f"Saved FDTD template to: {output_path}")
        print(f"Included pillar: {not args.reference}")
        print(f"Source polarization: {args.polarization_deg} deg")
    finally:
        fdtd.close()


if __name__ == "__main__":
    main()
