# PaFTR

## Getting Started

### 1. Environment Requirements

This project is implemented with PyTorch. Create a virtual environment and install dependencies.

#### Option A: `venv` (recommended)

Linux / macOS / WSL:

```bash
python3 -m venv PaFTR
source PaFTR/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -m venv PaFTR
.\PaFTR\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Option B: Conda

```bash
conda create -n PaFTR python=3.8
conda activate PaFTR
pip install -r requirements.txt
```

### 2. Prepare Data

Create `./dataset/` and place the files below (directly under `./dataset/`).

#### Public benchmarks used in the paper

| Dataset | File | Public source |
| -------- | ---- | ------------- |
| PEMS03/04/07/08 | `PEMS03.npz` ... `PEMS08.npz` | [Google Drive](https://drive.google.com/file/d/1bNbw1y8VYp-8pkRTqbjoW-TA-G8T0EQf/view) |
| Traffic | `traffic.csv` | [Google Drive](https://drive.google.com/file/d/1bNbw1y8VYp-8pkRTqbjoW-TA-G8T0EQf/view) |
| OD | `OD_2976.csv` | Not public |

This repo supports two dataset types (set by `--data`):

- **`custom`**: CSV file with a `date` column.
  - **Format**: `date, <feature_1>, ..., <feature_n>, <target>`
  - **Default target column name**: `OT` (set by `--target`)
  - **Example used in scripts**: `./dataset/traffic.csv`
- **`PEMS`**: `.npz` file containing a `data` array.
  - **Format**: `np.load(...)['data'][:, :, 0]` is used in code
  - **Example used in scripts**: `./dataset/PEMS03.npz`

**Note**: File path is controlled by `--root_path` and `--data_path`. For example, `--root_path ./dataset/ --data_path traffic.csv` expects `./dataset/traffic.csv`.

### 3. Training Example (Reproduce)

You can reproduce results by running the provided scripts (Git Bash / WSL / Linux).

#### Option A: Run all main experiments

```bash
sh run.sh
```

This will execute:

- `scripts/PaFTR/pems03.sh`
- `scripts/PaFTR/pems04.sh`
- `scripts/PaFTR/pems07.sh`
- `scripts/PaFTR/pems08.sh`
- `scripts/PaFTR/traffic.sh`

**Note**: `scripts/PaFTR/od.sh` is provided for reference but disabled in `run.sh` because the OD dataset (`OD_2976.csv`) is not publicly available.

#### Option B: Run a single dataset

For example:

```bash
sh scripts/PaFTR/pems03.sh
```

Each main script trains three random seeds (`2026 2027 2028`) and the four prediction lengths used in the paper. Main-table numbers are the mean ± std over the three seeds.

### 4. Ablation and Phase-Index Scripts

Ablations are under `scripts/Ablation/` (e.g., `scripts/Ablation/pems03.sh`) and use a single seed (`2026`).

They control ablation-related arguments in `run.py`, including:

- `--use_seq_cycle_complex` in `{seq, cycle, complex}`
- `--fusion_type` in `{freq, time_add, time_concat}`
- `--qkv` in `{cfs, csf, fcs, fsc, sfc, scf}`

Phase-index shift (frozen weights, test-only) is under `scripts/Phase/`. It loads the main seed-`2026` checkpoint and applies `--cycle_shift` in clock hours: ±12/24/36 steps on PEMS, ±4/8/12 on OD, ±1/2/3 on Traffic. Run it after the corresponding `scripts/PaFTR/` job has finished.

## Code Structure

- `run.py`: main entry (argument parsing + train/test loop)
- `exp/exp_main.py`: training / validation / testing pipeline
- `models/PaFTR.py`: PaFTR model implementation
- `data_provider/`: dataset loading and dataloader building
- `scripts/`: runnable experiment scripts (`PaFTR/`, `Ablation/`, `Phase/`)
