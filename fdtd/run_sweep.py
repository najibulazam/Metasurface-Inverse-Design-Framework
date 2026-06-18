"""
run_sweep.py

Orchestrates a geometry sweep over rectangular Si nanopillar unit cells,
reusing a single FDTD session (much faster than reopening per geometry --
session startup alone costs several seconds and a license checkout).

Produces:
    dataset/unit_cell_raw.csv
        columns: x_size_nm, y_size_nm, wavelength_nm, Txx_real, Txx_imag,
                 Tyy_real, Tyy_imag

This is the scripted equivalent of preprocess/FDTD_Simulation/unit_cell.py +
unit_script.lsf combined, but it actually runs the simulations instead of
just writing a parameter list and waiting for you to run Lumerical by hand.

Sizing note for your hardware (i7-14th gen 20-core, 64GB RAM, RTX 5070 Ti):
FDTD itself is CPU + RAM bound, not GPU bound, unless you have a GPU FDTD
solver license. A single unit cell at the mesh setting below typically
takes anywhere from ~10s to a few minutes depending on accuracy and size,
so start with the small grid below (5x5=25 geometries) to confirm
everything works before committing to a large grid (e.g. 30x30=900,
which could take hours).
"""

import argparse
import csv
import os
import time

from lumapi_setup.lumapi_loader import lumapi
from fdtd.unit_cell_sim import simulate_linear_unit_cell

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "unit_cell_raw.csv")


def parse_args():
    p = argparse.ArgumentParser(description="Sweep FDTD unit-cell geometries.")
    p.add_argument("--min_size", type=int, default=60, help="min pillar size (nm)")
    p.add_argument("--max_size", type=int, default=200, help="max pillar size (nm), exclusive of step boundary")
    p.add_argument("--step", type=int, default=40, help="step between sizes (nm)")
    p.add_argument("--height", type=int, default=450, help="pillar height (nm), fixed across sweep")
    p.add_argument("--start_wave", type=int, default=400, help="start wavelength (nm)")
    p.add_argument("--end_wave", type=int, default=800, help="stop wavelength (nm)")
    p.add_argument("--points", type=int, default=10, help="number of wavelength points")
    p.add_argument("--mesh_accuracy", type=int, default=4, help="Lumerical mesh accuracy setting")
    p.add_argument("--hide", action="store_true", help="hide the Lumerical GUI window while running")
    p.add_argument("--resume", action="store_true",
                   help="skip (x,y) pairs already present in the output CSV")
    return p.parse_args()


def already_done(csv_path):
    """Returns a set of (x_size, y_size) tuples already present in the CSV,
    so --resume can skip them after an interrupted run."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((int(row["x_size_nm"]), int(row["y_size_nm"])))
    return done


def main():
    args = parse_args()
    sizes = list(range(args.min_size, args.max_size, args.step))
    grid = [(x, y) for x in sizes for y in sizes]

    skip = already_done(OUTPUT_CSV) if args.resume else set()
    todo = [(x, y) for (x, y) in grid if (x, y) not in skip]

    print(f"Total geometries in grid: {len(grid)}")
    print(f"Already done (skipped):   {len(skip)}")
    print(f"Remaining to simulate:    {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    write_header = not os.path.exists(OUTPUT_CSV) or not args.resume
    mode = "w" if write_header else "a"

    fdtd = lumapi.FDTD(hide=args.hide)
    csv_file = open(OUTPUT_CSV, mode, newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(["x_size_nm", "y_size_nm", "wavelength_nm",
                         "Txx_real", "Txx_imag", "Tyy_real", "Tyy_imag"])

    try:
        for i, (x_size, y_size) in enumerate(todo):
            t0 = time.time()
            wavelengths_nm, txx, tyy = simulate_linear_unit_cell(
                fdtd,
                x_size,
                y_size,
                height_nm=args.height,
                lambda_start_nm=args.start_wave,
                lambda_stop_nm=args.end_wave,
                freq_points=args.points,
                mesh_accuracy=args.mesh_accuracy,
            )

            for wl, tx, ty in zip(wavelengths_nm, txx, tyy):
                writer.writerow([x_size, y_size, wl,
                                 tx.real, tx.imag, ty.real, ty.imag])
            csv_file.flush()

            elapsed = time.time() - t0
            print(f"[{i + 1}/{len(todo)}] x={x_size}nm y={y_size}nm "
                  f"done in {elapsed:.1f}s")
    finally:
        csv_file.close()
        fdtd.close()

    print(f"Sweep complete. Raw data written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
