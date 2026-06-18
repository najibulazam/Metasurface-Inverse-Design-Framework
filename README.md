# Metasurface Inverse Design Framework

An end-to-end research framework for **metasurface inverse design using physics-based simulation and deep learning**.

The pipeline integrates Ansys Lumerical FDTD unit-cell simulations, physics-informed data augmentation for Jones matrices, self-supervised Vision Transformer (ViT/SimMIM) pretraining, and direct structural parameter prediction (Inverse Design) optimized for modern NVIDIA GPUs.

---

## Features

- **Electromagnetic Simulation**: Automated rectangular Si nanopillar geometry sweeps using `lumapi` without relying on the Lumerical GUI.
- **Jones-Matrix Augmentation**: Mathematical rotation and pairing of single unit cells into full-parameter **Dimer** (supercell) responses.
- **Self-Supervised Pretraining**: Lightweight Vision Transformer (ViT) with SimMIM masked modeling to learn broadband electromagnetic representations.
- **Inverse Design Fine-Tuning**: One-shot structural parameter prediction ($x_1, y_1, \theta_1, x_2, y_2, \theta_2$) directly from an optical response target.
- **Hardware Optimized**: Fully tailored for **NVIDIA RTX 5070 Ti (Blackwell Architecture, sm_120)** using PyTorch Nightly and Native Mixed Precision (AMP) to prevent CPU fallback.

---

## Project Structure

```text
metasurface-project/
├─ config.yaml
├─ requirements.txt
├─ README.md
├─ lumapi_setup/
│  └─ lumapi_loader.py
├─ fdtd/
│  ├─ build_fdtd_template.py
│  ├─ run_sweep.py
│  └─ unit_cell_sim.py
├─ jones_matrix/
│  ├─ build_dataset.py
│  └─ jones_calc.py
├─ model/
│  ├─ simmim.py
│  └─ vit_small.py
├─ train/
│  ├─ dataset_loader.py
│  ├─ pretrain.py
│  └─ inverse_fine_tune.py
├─ design/
│  └─ metalens_design.py
├─ dataset/
└─ outputs/
   ├─ checkpoints/
   ├─ fdtd_templates/
   ├─ figures/
   └─ logs/

```

---

## Hardware & Environment Setup

### 1. Prerequisites

* **OS**: Windows 11
* **CPU**: Intel Core i7 14th Generation
* **RAM**: 64GB
* **Simulation**: Ansys Lumerical FDTD v241 (with Python API configured)

### 2. PyTorch Installation (Optimized for RTX 5070 Ti)

To fully utilize the Compute Capability 12.0 of the Blackwell GPU and prevent CPU fallback, install the CUDA 13 compatible PyTorch Nightly build:

```powershell
# Uninstall old torch distributions if any
pip uninstall torch torchvision torchaudio -y

# Install PyTorch Nightly with CUDA 13 support
pip install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu130](https://download.pytorch.org/whl/nightly/cu130)
pip install -r requirements.txt

```

### 3. Verify GPU Detection

Run the following check to confirm that PyTorch successfully claims your GPU:

```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

```

**Expected Output:**

```text
CUDA Available: True
Device Name: NVIDIA GeForce RTX 5070 Ti

```

---

## Typical Workflow

The entire pipeline must be executed in the following chronological order:

### 1. Build FDTD Unit-Cell Template

Generate a sample `.fsp` file to verify materials, monitors, boundary conditions, and geometry setup in the Lumerical GUI.

```powershell
python -m fdtd.build_fdtd_template --x_size 120 --y_size 80 --height 450

```

### 2. Run Geometry Sweep

Execute automated x- and y-polarization simulations across a grid of sizes to harvest broadband raw transmission data ($T_{xx}$ and $T_{yy}$).

```powershell
python -m fdtd.run_sweep --min_size 80 --max_size 161 --step 40 --points 11 --hide

```

### 3. Build Jones-Matrix Dataset

Perform physics-informed data augmentation by rotating single cells and pairing them into complex dimers. This exports the normalized CSV datasets.

```powershell
python -m jones_matrix.build_dataset --angle_step 20 --max_pairs 20000

```

### 4. Self-Supervised Pretraining (SimMIM)

Pretrain the Vision Transformer (ViT) on your GPU using masked Jones-matrix reconstruction. Uses AMP for high-throughput training.

```powershell
python -m train.pretrain --epochs 50 --batch_size 256

```

### 5. Inverse Design Fine-Tuning

Fine-tune the pretrained ViT encoder with a multi-layer prediction head to map full-wavelength Jones matrices directly to 6 structural dimensions.

```powershell
python -m train.inverse_fine_tune

```

### 6. Generate Metalens Target

Compute the ideal broadband hyperbolic phase profile across a $64 \times 64$ grid to prepare the ultimate validation target for your inverse design model.

```powershell
python -m design.metalens_design --size 64 --focus_um 75

```

---

## Notes & Best Practices

* **FDTD Benchmarking**: Electromagnetic simulation is heavily CPU/RAM bound (unless utilizing an explicit GPU FDTD license). Always run a minimal sweep grid (e.g., $5 \times 5$) to calculate expected execution time before launching large matrix sweeps.
* **Data Loading Speed**: The `dataset_loader.py` is configured with `pin_memory=True` and `num_workers=4` to streamline batch processing, keeping the high-speed RTX 5070 Ti fully saturated.
* **Lumerical Integration**: `lumapi` comes bundled with Ansys Lumerical and is loaded dynamically via `lumapi_setup/lumapi_loader.py`. Ensure your Lumerical installation matches the path defined in the loader.

---

## Citation

If you use this framework or parts of this codebase in academic publications, please cite the corresponding references:

1. Yan et al., *Metasurface Vision Transformer: A Generic AI Model for Metasurface Inverse Design*, Nanophotonics, 2026.
2. He et al., *Masked Autoencoders Are Scalable Vision Learners* (SimMIM equivalents).