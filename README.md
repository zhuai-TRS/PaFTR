# PaFTR

## Getting Started

### 1、Environment Requirements

This project is implemented with PyTorch. Create a virtual environment and install dependencies.

#### Option A: `venv` (recommended)

Linux / macOS / WSL:

```
python3 -m venv PaFTR
source PaFTR/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```
py -m venv PaFTR
.\PaFTR\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Option B: Conda

```
conda create -n PaFTR python=3.8
conda activate PaFTR
pip install -r requirements.txt
```

### 2、Prepare Data

**Download the datasets first** (PEMS and traffic), then place them under `./dataset/`.


| Source        | Link                                                                                                                                                                               |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baidu Netdisk | [https://pan.baidu.com/s/18qiqVq__shRYWFNOuOblHg?pwd=qf4t](https://pan.baidu.com/s/18qiqVq__shRYWFNOuOblHg?pwd=qf4t) (extract code: `qf4t`)                                        |
| Google Drive  | [https://drive.google.com/drive/folders/1Rqrljn3XA4hFmwilmH_BdZq65eyY03MZ?usp=drive_link](https://drive.google.com/drive/folders/1Rqrljn3XA4hFmwilmH_BdZq65eyY03MZ?usp=drive_link) |


Create a folder named `./dataset` and put the downloaded files into this directory (directly under `./dataset/`).

This repo supports two dataset types (set by `--data`):

- `**custom`**: CSV file with a `date` column.
  - **Format**: `date, <feature_1>, ..., <feature_n>, <target>`
  - **Default target column name**: `OT` (set by `--target`)
  - **Example used in scripts**:
    - `./dataset/traffic.csv`
- `**PEMS`**: `.npz` file containing `data` array.
  - **Format**: `np.load(... )['data'][:, :, 0]` is used in code
  - **Example used in scripts**:
    - `./dataset/PEMS03.npz`

**Note**: File path is controlled by `--root_path` and `--data_path`. For example, `--root_path ./dataset/ --data_path traffic.csv` expects `./dataset/traffic.csv`.

### 3、Training Example (Reproduce)

You can reproduce results by running the provided scripts.

#### Option A: Run all main experiments

```
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

```
sh scripts/PaFTR/pems03.sh
```

Each script runs multiple prediction lengths in a loop by calling `run.py`.

### 4、Ablation Scripts

Ablations are provided under `scripts/Ablation/` (e.g., `scripts/Ablation/pems03.sh`).

They control ablation-related arguments in `run.py`, including:

- `--use_seq_cycle_complex` in `{seq, cycle, complex}`
- `--fusion_type` in `{freq, time_add, time_concat}`
- `--qkv` in `{cfs, csf, fcs, fsc, sfc, scf}`

## Code Structure

- `run.py`: main entry (argument parsing + train/test loop)
- `exp/exp_main.py`: training/validation/testing pipeline
- `models/PaFTR.py`: PaFTR model implementation
- `data_provider/`: dataset loading and dataloader building
- `scripts/`: runnable experiment scripts

