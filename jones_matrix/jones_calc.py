"""
jones_calc.py

Converts raw unit-cell FDTD field data (Ex, Ey complex transmission per
geometry per wavelength) into Jones matrices, applies in-plane rotation,
and combines two rotated units into a "dimer" -- mirroring the math in
preprocess/Jones_matrix_calculation/{jones_matrix,jones_vector,double_cell}.py
from the original repo, but written as plain functions (no custom classes)
operating directly on numpy arrays for simplicity.

Physics recap:
    A rectangular Si pillar acts as a linear waveplate: light polarized
    along its long axis (Ax) and short axis (Ay) picks up different
    amplitude/phase. In the pillar's own (x,y) frame the Jones matrix is
    diagonal:
        J_unit = [[Ax*exp(i*phix), 0      ],
                  [0,        Ay*exp(i*phiy)]]
    Rotating the pillar by angle theta in the lab frame:
        J_rot(theta) = R(-theta) @ J_unit @ R(theta)
    A "dimer" (two pillars per unit cell, as in the repo) just adds the two
    rotated Jones matrices -- this models superposing the two structures'
    contributions within one supercell period.
"""

import numpy as np


def rotation_matrix(theta_deg, shape):
    """2x2 rotation matrices, batched over `shape` (e.g. one per wavelength).
    Returns an array of shape (*shape, 2, 2).
    """
    theta = np.radians(theta_deg)
    cos_t = np.cos(theta) * np.ones(shape)
    sin_t = np.sin(theta) * np.ones(shape)
    R = np.stack([
        np.stack([cos_t, -sin_t], axis=-1),
        np.stack([sin_t, cos_t], axis=-1),
    ], axis=-2)
    return R  # shape (*shape, 2, 2)


def unit_jones_matrix(Ax, Ay, phix, phiy):
    """Builds the diagonal Jones matrix of an unrotated unit cell.
    Ax, Ay, phix, phiy: arrays of shape (n_wave,) -- one value per wavelength.
    Returns array of shape (n_wave, 2, 2), complex.
    """
    n_wave = Ax.shape[0]
    J = np.zeros((n_wave, 2, 2), dtype=complex)
    J[:, 0, 0] = Ax * np.exp(1j * phix)
    J[:, 1, 1] = Ay * np.exp(1j * phiy)
    return J


def rotate_jones_matrix(J_unit, theta_deg):
    """Applies J_rot = R(-theta) @ J_unit @ R(theta).
    J_unit: (n_wave, 2, 2) complex
    Returns: (n_wave, 2, 2) complex
    """
    n_wave = J_unit.shape[0]
    R_minus = rotation_matrix(-theta_deg, (n_wave,))
    R_plus = rotation_matrix(theta_deg, (n_wave,))
    return R_minus @ J_unit @ R_plus


def combine_dimer(J_a, J_b):
    """Combines two rotated unit Jones matrices into one dimer response.
    Simple superposition, matching double_cell.py's `unit_JM[partA] + unit_JM[partB]`.
    """
    return J_a + J_b


def jones_to_amp_phase_vector(J):
    """Flattens a (n_wave, 2, 2) complex Jones matrix stack into the repo's
    6-channel convention per wavelength: [A11, A12, A22, phi11, phi12, phi22]
    (J21 is dropped since for these reciprocal structures J21 == J12).

    Returns array of shape (n_wave, 6).
    """
    n_wave = J.shape[0]
    out = np.zeros((n_wave, 6))
    out[:, 0] = np.abs(J[:, 0, 0])       # A11
    out[:, 1] = np.abs(J[:, 0, 1])       # A12
    out[:, 2] = np.abs(J[:, 1, 1])       # A22
    out[:, 3] = np.angle(J[:, 0, 0])     # phi11
    out[:, 4] = np.angle(J[:, 0, 1])     # phi12
    out[:, 5] = np.angle(J[:, 1, 1])     # phi22
    return out


def normalize_amp_phase(JM_vec, amp_cols=(0, 1, 2), phase_cols=(3, 4, 5)):
    """Per-column min-max normalize amplitudes to [-1, 1] and phases to
    [-1, 1] (representing -pi..pi), matching the repo's training
    convention. Operates on a full dataset array of shape (N, 6) so the
    min/max are dataset-wide, not per-sample.

    Returns: normalized array (N, 6), and a dict of the min/max used (so
    you can invert the normalization later, or replicate the repo's
    min_max_mean_list.txt for downstream stats-based bias adjustment).
    """
    JM_norm = JM_vec.copy()
    stats = {}
    for col in amp_cols:
        c_min, c_max = JM_vec[:, col].min(), JM_vec[:, col].max()
        JM_norm[:, col] = 2 * (JM_vec[:, col] - c_min) / (c_max - c_min + 1e-12) - 1
        stats[col] = (c_min, c_max)
    for col in phase_cols:
        # phase is already bounded in [-pi, pi] by np.angle, so a fixed
        # rescale (not data-dependent min/max) keeps things consistent
        # across different sweeps/datasets.
        JM_norm[:, col] = JM_vec[:, col] / np.pi
        stats[col] = (-np.pi, np.pi)
    return JM_norm, stats