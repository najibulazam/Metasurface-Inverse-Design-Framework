"""
metalens_design.py

Builds the "design target" for a broadband achromatic-style metalens:
for each wavelength, compute the ideal hyperbolic phase profile across a
2D array of unit-cell positions, mirroring evaluation/metasurface_design/
utils.py's LensPhase() function and JM_generator.py's JM_type4 path from
the original repo.

This produces a target Jones-matrix-like array [size, size, n_wave, 6]
where the diagonal phase channels encode the focusing phase profile.
The cross-polar channel (A12 / phi12) is left unconstrained by default,
because the training dataset only guarantees that off-diagonal terms
emerge from rotated anisotropic cells; a scalar metalens target does not
generally require a specific cross-polar phase.

Output:
    dataset/metalens_target_JM.npy   shape [size, size, n_wave, 6]
    dataset/metalens_target_mask.npy same shape, 0 = constrained (phase), 1 = free

What you'd do with this next (not automated here, since it depends on
your fine-tuned model from train/pretrain.py + a fine-tuning step you
haven't built yet): feed this target through your fine-tuned model to
predict per-pixel structural parameters (pillar x/y/angle for two units),
then export those into a Lumerical construction script similar to the
original repo's metalens_output/lens_construct.lsf to actually build and
verify the lens in FDTD.
"""

import argparse
import os

import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")


def lens_phase(size_x, size_y, unit_x_m, unit_y_m, focus_length_m, wavelength_m):
    """Ideal hyperbolic lens phase profile, identical formula to the
    repo's evaluation/metasurface_design/utils.py LensPhase(), minus the
    plotting/visualize branch.

    Returns phase array [size_x, size_y] wrapped to [-pi, pi].
    """
    x = (np.arange(-size_x / 2, size_x / 2)) * unit_x_m
    y = (np.arange(-size_y / 2, size_y / 2)) * unit_y_m
    X, Y = np.meshgrid(x, y, indexing="ij")
    phase = (2 * np.pi / wavelength_m) * (focus_length_m - np.sqrt(focus_length_m ** 2 + X ** 2 + Y ** 2))
    return np.angle(np.exp(1j * phase))


def build_metalens_target(size, wavelengths_nm, focus_length_um,
                          unit_x_nm, unit_y_nm, amplitude_mode="all"):
    """Builds the target JM array and corresponding mask.

    amplitude_mode:
        "all"  -> amplitudes unconstrained everywhere (mask=1 for A11,A12,A22)
                  while diagonal phases phi11 / phi22 are constrained.
        "none" -> same, but explicit alias kept for clarity/parity with repo args.
    """
    n_wave = len(wavelengths_nm)
    JM = np.zeros((size, size, n_wave, 6))
    mask = np.ones((size, size, n_wave, 6))  # 1 = free/unconstrained by default

    unit_x_m = unit_x_nm * 1e-9
    unit_y_m = unit_y_nm * 1e-9
    focus_length_m = focus_length_um * 1e-6

    for i, wl_nm in enumerate(wavelengths_nm):
        wl_m = wl_nm * 1e-9
        phase = lens_phase(size, size, unit_x_m, unit_y_m, focus_length_m, wl_m)
        phase_norm = phase / np.pi  # rescale to [-1, 1] to match training convention

        JM[:, :, i, 3] = phase_norm  # phi11
        JM[:, :, i, 5] = phase_norm  # phi22

        # A scalar/broadband metalens target constrains the diagonal phase
        # response. The cross-polar channel remains free unless a more
        # specialized polarization-transforming device is being designed.
        mask[:, :, i, 3] = 0
        mask[:, :, i, 5] = 0

    return JM, mask


def main():
    p = argparse.ArgumentParser(description="Build a target Jones-matrix array for a broadband metalens.")
    p.add_argument("--size", type=int, default=64, help="lens array side length in unit cells (start small)")
    p.add_argument("--focus_um", type=float, default=75.0, help="focal length in micrometers")
    p.add_argument("--unit_x_nm", type=float, default=500.0, help="unit-cell pitch x (nm)")
    p.add_argument("--unit_y_nm", type=float, default=250.0, help="unit-cell pitch y (nm)")
    p.add_argument("--wavelengths_csv", default=os.path.join(DATASET_DIR, "wavelengths_nm.csv"),
                   help="CSV with a wavelength_nm column, from build_dataset.py. "
                        "If missing, falls back to a default 5-point sweep.")
    args = p.parse_args()

    if os.path.exists(args.wavelengths_csv):
        import pandas as pd
        wavelengths_nm = pd.read_csv(args.wavelengths_csv)["wavelength_nm"].to_numpy()
    else:
        print(f"Warning: {args.wavelengths_csv} not found, using a default 5-point 450-650nm sweep.")
        wavelengths_nm = np.linspace(450, 650, 5)

    JM, mask = build_metalens_target(
        args.size, wavelengths_nm, args.focus_um, args.unit_x_nm, args.unit_y_nm
    )

    jm_path = os.path.join(DATASET_DIR, "metalens_target_JM.npy")
    mask_path = os.path.join(DATASET_DIR, "metalens_target_mask.npy")
    np.save(jm_path, JM)
    np.save(mask_path, mask)

    print(f"Built metalens target: size={args.size}x{args.size}, "
         f"n_wave={len(wavelengths_nm)}, focus={args.focus_um}um")
    print(f"Saved target JM to: {jm_path}")
    print(f"Saved target mask to: {mask_path}")
    print("Diagonal phase channels phi11 and phi22 are constrained (mask=0); "
         "amplitudes and the cross-polar channel remain free (mask=1).")


if __name__ == "__main__":
    main()
