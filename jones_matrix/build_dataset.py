"""
build_dataset.py

Takes the raw unit-cell CSV produced by fdtd/run_sweep.py and builds the
final ViT-ready dataset:

    1. Load raw (x_size, y_size, wavelength, Txx, Tyy) rows.
    2. Treat the normalized co-polar transmission coefficients Txx and Tyy
       as the unrotated anisotropic unit-cell response.
    3. Build the unrotated Jones matrix per geometry per wavelength.
    4. Rotate each unit cell through a set of angles (0 to 80 deg, step
       configurable) -- mirrors double_cell.py's angle sweep.
    5. Pair up every (rotated unit A, rotated unit B) combination into a
       "dimer" by summing their Jones matrices -- mirrors JM_double().
    6. Flatten to the 6-channel [A11,A12,A22,phi11,phi12,phi22] convention,
       normalize, and write out:
         dataset/param_dimer.csv   columns: x1,y1,angle1,x2,y2,angle2
         dataset/JM_dimer.csv      columns: wave0_A11 ... wave{N-1}_phi22
         dataset/dataset_stats.json   normalization min/max per channel

This is a CPU/RAM-bound combinatorial step (no FDTD calls), and is exactly
why your 64GB RAM and 20 cores matter: the dimer pairing is O(n^2) in the
number of (geometry x angle) units, so keep angle_step coarse-ish (e.g. 20
deg) and your size grid modest for a first pass, then scale up once you've
validated the pipeline end-to-end.
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from jones_matrix.jones_calc import (
    unit_jones_matrix,
    rotate_jones_matrix,
    combine_dimer,
    jones_to_amp_phase_vector,
    normalize_amp_phase,
)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")


def load_raw(csv_path):
    """Loads the raw FDTD CSV and reshapes into per-geometry arrays.

    Returns:
        geometries: list of (x_size, y_size) tuples
        wavelengths_nm: array (n_wave,) -- assumed identical across all geometries
        Ax_table: dict (x_size, y_size) -> array (n_wave,) complex   [from Txx]
        Ay_table: dict (x_size, y_size) -> array (n_wave,) complex   [from Tyy]
    """
    df = pd.read_csv(csv_path)
    required_cols = {
        "x_size_nm", "y_size_nm", "wavelength_nm",
        "Txx_real", "Txx_imag", "Tyy_real", "Tyy_imag",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            "Raw CSV is missing required columns: "
            f"{sorted(missing)}. Re-run fdtd/run_sweep.py with the current code."
        )
    wavelengths_nm = np.sort(df["wavelength_nm"].unique())
    geometries = sorted(set(zip(df["x_size_nm"], df["y_size_nm"])))

    Ax_table, Ay_table = {}, {}
    for (x, y) in geometries:
        sub = df[(df["x_size_nm"] == x) & (df["y_size_nm"] == y)].sort_values("wavelength_nm")
        assert len(sub) == len(wavelengths_nm), (
            f"Geometry ({x},{y}) has {len(sub)} wavelength rows, expected {len(wavelengths_nm)}. "
            "Sweep may be incomplete -- check run_sweep.py output."
        )
        txx = sub["Txx_real"].to_numpy() + 1j * sub["Txx_imag"].to_numpy()
        tyy = sub["Tyy_real"].to_numpy() + 1j * sub["Tyy_imag"].to_numpy()
        Ax_table[(x, y)] = txx
        Ay_table[(x, y)] = tyy

    return geometries, wavelengths_nm, Ax_table, Ay_table


def build_rotated_units(geometries, Ax_table, Ay_table, angles_deg):
    """For every (geometry, angle) pair, builds the rotated Jones matrix.

    Returns:
        unit_params: list of (x_size, y_size, angle) -- parallel to unit_JMs
        unit_JMs: list of (n_wave, 2, 2) complex arrays
    """
    unit_params = []
    unit_JMs = []
    for (x, y) in geometries:
        Ax, Ay = Ax_table[(x, y)], Ay_table[(x, y)]
        amp_x, phi_x = np.abs(Ax), np.angle(Ax)
        amp_y, phi_y = np.abs(Ay), np.angle(Ay)
        J0 = unit_jones_matrix(amp_x, amp_y, phi_x, phi_y)
        for theta in angles_deg:
            J_rot = rotate_jones_matrix(J0, theta)
            unit_params.append((x, y, theta))
            unit_JMs.append(J_rot)
    return unit_params, unit_JMs


def build_dimers(unit_params, unit_JMs, max_pairs=None, seed=0):
    """Pairs every unit with every other unit (including itself), summing
    their Jones matrices, mirroring JM_double_pretrained's upper-triangle
    pairing (partA, partB) with partB >= partA.

    If max_pairs is set, randomly subsamples down to that many pairs
    instead of the full combinatorial set -- use this once your grid is
    larger than you want to fully expand (n choose 2 grows fast).

    Returns:
        dimer_params: array (n_dimer, 6) -- [x1,y1,angle1,x2,y2,angle2]
        dimer_JMs: array (n_dimer, n_wave, 2, 2) complex
    """
    n = len(unit_params)
    pair_indices = list(itertools.combinations_with_replacement(range(n), 2))

    if max_pairs is not None and max_pairs < len(pair_indices):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(pair_indices), size=max_pairs, replace=False)
        pair_indices = [pair_indices[i] for i in chosen]

    dimer_params = np.zeros((len(pair_indices), 6))
    n_wave = unit_JMs[0].shape[0]
    dimer_JMs = np.zeros((len(pair_indices), n_wave, 2, 2), dtype=complex)

    for i, (a, b) in enumerate(pair_indices):
        dimer_params[i, :3] = unit_params[a]
        dimer_params[i, 3:] = unit_params[b]
        dimer_JMs[i] = combine_dimer(unit_JMs[a], unit_JMs[b])

    return dimer_params, dimer_JMs


def flatten_and_normalize(dimer_JMs):
    """Converts (n_dimer, n_wave, 2, 2) complex -> (n_dimer, n_wave*6)
    normalized real-valued array, ready for the ViT.
    """
    n_dimer, n_wave = dimer_JMs.shape[0], dimer_JMs.shape[1]
    flat = np.zeros((n_dimer, n_wave, 6))
    for i in range(n_dimer):
        flat[i] = jones_to_amp_phase_vector(dimer_JMs[i])

    flat_2d = flat.reshape(-1, 6)  # stack all (sample, wave) rows for global normalization
    norm_2d, stats = normalize_amp_phase(flat_2d)
    norm = norm_2d.reshape(n_dimer, n_wave, 6)
    norm_flat = norm.reshape(n_dimer, n_wave * 6)
    return norm_flat, stats


def main():
    p = argparse.ArgumentParser(description="Build dimer Jones-matrix dataset from raw FDTD sweep.")
    p.add_argument("--raw_csv", default=os.path.join(DATASET_DIR, "unit_cell_raw.csv"))
    p.add_argument("--angle_step", type=int, default=20, help="rotation step in degrees, 0-80 range")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="cap total number of dimer samples (random subsample). "
                        "Leave unset to use the full combinatorial set.")
    p.add_argument("--out_prefix", default="dimer", help="prefix for output files")
    args = p.parse_args()

    geometries, wavelengths_nm, Ax_table, Ay_table = load_raw(args.raw_csv)
    print(f"Loaded {len(geometries)} geometries x {len(wavelengths_nm)} wavelengths from raw CSV.")

    angles_deg = list(range(0, 90, args.angle_step))
    unit_params, unit_JMs = build_rotated_units(geometries, Ax_table, Ay_table, angles_deg)
    print(f"Built {len(unit_params)} rotated units ({len(geometries)} geoms x {len(angles_deg)} angles).")

    dimer_params, dimer_JMs = build_dimers(unit_params, unit_JMs, max_pairs=args.max_pairs)
    print(f"Built {dimer_params.shape[0]} dimer samples.")

    norm_flat, stats = flatten_and_normalize(dimer_JMs)

    param_path = os.path.join(DATASET_DIR, f"param_{args.out_prefix}.csv")
    jm_path = os.path.join(DATASET_DIR, f"JM_{args.out_prefix}.csv")
    stats_path = os.path.join(DATASET_DIR, "dataset_stats.json")
    wave_path = os.path.join(DATASET_DIR, "wavelengths_nm.csv")

    pd.DataFrame(dimer_params, columns=["x1_nm", "y1_nm", "angle1_deg",
                                        "x2_nm", "y2_nm", "angle2_deg"]).to_csv(param_path, index=False)

    n_wave = len(wavelengths_nm)
    jm_cols = []
    for w in range(n_wave):
        jm_cols += [f"w{w}_A11", f"w{w}_A12", f"w{w}_A22",
                   f"w{w}_phi11", f"w{w}_phi12", f"w{w}_phi22"]
    pd.DataFrame(norm_flat, columns=jm_cols).to_csv(jm_path, index=False)

    pd.DataFrame({"wavelength_nm": wavelengths_nm}).to_csv(wave_path, index=False)

    with open(stats_path, "w") as f:
        json.dump({str(k): v for k, v in stats.items()}, f, indent=2)

    print(f"Wrote parameters to: {param_path}")
    print(f"Wrote Jones matrices to: {jm_path}")
    print(f"Wrote wavelength list to: {wave_path}")
    print(f"Wrote normalization stats to: {stats_path}")


if __name__ == "__main__":
    main()
