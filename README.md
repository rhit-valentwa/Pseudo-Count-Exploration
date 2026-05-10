# Pseudo-Count Exploration

Pseudo-count intrinsic exploration bonuses for tabular Q-learning, based on [Bellemare et al. 2016](https://arxiv.org/abs/1606.01898).

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```
or
```bash
python3 -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`

### 2. Install packages

```bash
pip install torch numpy matplotlib "gymnasium[atari]" "gymnasium[other]" ale-py autorom --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
```

### 3. Download Atari ROMs

The `AutoROM` command may not be on PATH after install. Use this instead:

```bash
python -c "from AutoROM.AutoROM import cli; from click.testing import CliRunner; CliRunner().invoke(cli, ['--accept-license'])"
```

## Run

```bash
python atari_dqn.py
python atari_dqn.py --env ALE/Breakout-v5 --steps 5000000
```

Graphs are saved to `graphs/`.

## Troubleshooting

**Script appears to hang on startup**
The replay buffer is pre-allocated in RAM. On the first run this can take 10–30 seconds before any output appears. The first log line prints at step 10,000 — wait for it before assuming a stall.


**`Namespace ALE not found`**
Verify `ale-py` is installed in the active venv:
```bash
python -c "import ale_py; print(ale_py.__version__)"
```

**`No module named AutoROM`**
The package name is case-sensitive. Confirm with:
```bash
python -c "import AutoROM; print('ok')"
```
If it fails, run `pip install autorom` and retry.

**CUDA not available**
The script falls back to CPU automatically (`Device: cpu`). Check your build:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, reinstall PyTorch with the correct CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).
