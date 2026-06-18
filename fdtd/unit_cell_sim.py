"""
unit_cell_sim.py

Builds and runs a 3D Lumerical FDTD unit-cell simulation for one
rectangular silicon nanopillar on a glass substrate.

The key requirement for a physically meaningful Jones matrix is that the
co-polarized transmission coefficients come from separate linear-input
simulations:

    Txx: x-polarized output under x-polarized illumination
    Tyy: y-polarized output under y-polarized illumination

This file therefore supports:
    1. building the device or a matching reference cell,
    2. running x/y polarization cases,
    3. normalizing the transmitted field against the corresponding
       reference simulation so the dataset stores transmission
       coefficients instead of raw monitor fields.
"""

import numpy as np

from lumapi_setup.lumapi_loader import lumapi

NM = 1e-9


def build_unit_cell(
    fdtd,
    x_size_nm,
    y_size_nm,
    height_nm,
    lambda_start_nm,
    lambda_stop_nm,
    freq_points,
    mesh_accuracy=4,
    source_polarization_deg=0.0,
    include_pillar=True,
    cell_margin_nm=300,
):
    """Construct the FDTD scene in the currently open session."""
    fdtd.switchtolayout()
    fdtd.selectall()
    fdtd.deleteall()

    height = height_nm * NM
    x_span = x_size_nm * NM
    y_span = y_size_nm * NM
    cell_x_span = (x_size_nm + cell_margin_nm) * NM
    cell_y_span = (y_size_nm + cell_margin_nm) * NM
    lambda_start = lambda_start_nm * NM
    lambda_stop = lambda_stop_nm * NM

    fdtd.addrect()
    fdtd.set("name", "substrate")
    fdtd.set("material", "SiO2 (Glass) - Palik")
    fdtd.set("x", 0)
    fdtd.set("y", 0)
    fdtd.set("x span", cell_x_span)
    fdtd.set("y span", cell_y_span)
    fdtd.set("z max", 0)
    fdtd.set("z min", -2 * lambda_stop)

    if include_pillar:
        fdtd.addrect()
        fdtd.set("name", "pillar")
        fdtd.set("material", "Si (Silicon) - Palik")
        fdtd.set("x", 0)
        fdtd.set("y", 0)
        fdtd.set("z min", 0)
        fdtd.set("z max", height)
        fdtd.set("x span", x_span)
        fdtd.set("y span", y_span)

    fdtd.addfdtd()
    fdtd.set("dimension", "3D")
    fdtd.set("x", 0)
    fdtd.set("y", 0)
    fdtd.set("x span", cell_x_span)
    fdtd.set("y span", cell_y_span)
    fdtd.set("z max", 0.5 * lambda_stop + height)
    fdtd.set("z min", -0.5 * lambda_stop)
    fdtd.set("x min bc", "Periodic")
    fdtd.set("y min bc", "Periodic")
    fdtd.set("mesh accuracy", mesh_accuracy)

    if include_pillar:
        fdtd.addmesh()
        fdtd.set("name", "pillar_mesh")
        fdtd.set("x", 0)
        fdtd.set("y", 0)
        fdtd.set("x span", x_span)
        fdtd.set("y span", y_span)
        fdtd.set("z min", 0)
        fdtd.set("z max", height)
        fdtd.set("dx", max(x_span / 30, 5 * NM))
        fdtd.set("dy", max(y_span / 30, 5 * NM))
        fdtd.set("dz", max(height / 30, 5 * NM))

    fdtd.addplane()
    fdtd.set("name", "source")
    fdtd.set("injection axis", "z")
    fdtd.set("direction", "Forward")
    fdtd.set("x", 0)
    fdtd.set("y", 0)
    fdtd.set("x span", cell_x_span * 1.5)
    fdtd.set("y span", cell_y_span * 1.5)
    fdtd.set("z", -0.4 * lambda_stop)
    fdtd.set("wavelength start", lambda_start)
    fdtd.set("wavelength stop", lambda_stop)
    fdtd.set("polarization angle", float(source_polarization_deg))

    fdtd.setglobalmonitor("frequency points", freq_points)

    fdtd.addpower()
    fdtd.set("name", "trans")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x", 0)
    fdtd.set("y", 0)
    fdtd.set("x span", cell_x_span)
    fdtd.set("y span", cell_y_span)
    fdtd.set("z", 0.4 * lambda_stop + height)


def run_and_extract_fields(fdtd):
    """Run the current scene and return spatially averaged complex fields."""
    fdtd.save("unit_cell_temp.fsp")
    fdtd.run()

    field = fdtd.getresult("trans", "E")
    ex = field["E"][:, :, :, :, 0]
    ey = field["E"][:, :, :, :, 1]

    ex_avg = np.mean(ex, axis=(0, 1, 2))
    ey_avg = np.mean(ey, axis=(0, 1, 2))
    wavelengths_nm = np.squeeze(field["lambda"]) * 1e9

    fdtd.switchtolayout()
    return wavelengths_nm, ex_avg, ey_avg


def _safe_complex_divide(numerator, denominator, eps=1e-12):
    out = np.zeros_like(numerator, dtype=complex)
    valid = np.abs(denominator) > eps
    out[valid] = numerator[valid] / denominator[valid]
    return out


def simulate_polarized_fields(
    fdtd,
    x_size_nm,
    y_size_nm,
    height_nm,
    lambda_start_nm,
    lambda_stop_nm,
    freq_points,
    source_polarization_deg,
    include_pillar=True,
    mesh_accuracy=4,
):
    """Build, run, and extract fields for one input polarization."""
    build_unit_cell(
        fdtd,
        x_size_nm,
        y_size_nm,
        height_nm,
        lambda_start_nm,
        lambda_stop_nm,
        freq_points,
        mesh_accuracy=mesh_accuracy,
        source_polarization_deg=source_polarization_deg,
        include_pillar=include_pillar,
    )
    return run_and_extract_fields(fdtd)


def simulate_linear_unit_cell(
    fdtd,
    x_size_nm,
    y_size_nm,
    height_nm=450,
    lambda_start_nm=400,
    lambda_stop_nm=800,
    freq_points=20,
    mesh_accuracy=4,
):
    """Return normalized co-polar transmission for x- and y-input states."""
    wl_x_ref, ex_ref, _ = simulate_polarized_fields(
        fdtd,
        x_size_nm,
        y_size_nm,
        height_nm,
        lambda_start_nm,
        lambda_stop_nm,
        freq_points,
        source_polarization_deg=0.0,
        include_pillar=False,
        mesh_accuracy=mesh_accuracy,
    )
    wl_x, ex_dev, _ = simulate_polarized_fields(
        fdtd,
        x_size_nm,
        y_size_nm,
        height_nm,
        lambda_start_nm,
        lambda_stop_nm,
        freq_points,
        source_polarization_deg=0.0,
        include_pillar=True,
        mesh_accuracy=mesh_accuracy,
    )
    wl_y_ref, _, ey_ref = simulate_polarized_fields(
        fdtd,
        x_size_nm,
        y_size_nm,
        height_nm,
        lambda_start_nm,
        lambda_stop_nm,
        freq_points,
        source_polarization_deg=90.0,
        include_pillar=False,
        mesh_accuracy=mesh_accuracy,
    )
    wl_y, _, ey_dev = simulate_polarized_fields(
        fdtd,
        x_size_nm,
        y_size_nm,
        height_nm,
        lambda_start_nm,
        lambda_stop_nm,
        freq_points,
        source_polarization_deg=90.0,
        include_pillar=True,
        mesh_accuracy=mesh_accuracy,
    )

    if not (np.allclose(wl_x_ref, wl_x) and np.allclose(wl_x, wl_y_ref) and np.allclose(wl_y_ref, wl_y)):
        raise RuntimeError("Wavelength grids do not match across polarization/reference runs.")

    txx = _safe_complex_divide(ex_dev, ex_ref)
    tyy = _safe_complex_divide(ey_dev, ey_ref)
    return wl_x, txx, tyy


def simulate_one_unit_cell(
    x_size_nm,
    y_size_nm,
    height_nm=450,
    lambda_start_nm=400,
    lambda_stop_nm=800,
    freq_points=20,
    hide=True,
    mesh_accuracy=4,
):
    """Open one session, run both polarizations plus references, then close."""
    fdtd = lumapi.FDTD(hide=hide)
    try:
        wavelengths_nm, txx, tyy = simulate_linear_unit_cell(
            fdtd,
            x_size_nm,
            y_size_nm,
            height_nm=height_nm,
            lambda_start_nm=lambda_start_nm,
            lambda_stop_nm=lambda_stop_nm,
            freq_points=freq_points,
            mesh_accuracy=mesh_accuracy,
        )
    finally:
        fdtd.close()
    return wavelengths_nm, txx, tyy


if __name__ == "__main__":
    wl, txx, tyy = simulate_one_unit_cell(x_size_nm=120, y_size_nm=80, hide=False)
    print("wavelengths (nm):", wl)
    print("Txx amplitude:", np.abs(txx))
    print("Txx phase (rad):", np.angle(txx))
    print("Tyy amplitude:", np.abs(tyy))
    print("Tyy phase (rad):", np.angle(tyy))
