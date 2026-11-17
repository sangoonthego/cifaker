# Contributing to ProofLens

Welcome to **ProofLens** (*Research and Development of an AI-Generated Image Detection Model*). This document defines the engineering standards, module ownership, branching strategy, code review checklist, and reproducibility rules for our four-member research team.

---

## 1. Team Ownership & Review Responsibilities

Ownership indicates primary architectural stewardship and pull-request review responsibility, not exclusive editing restrictions.

| Role | Owned Modules & Directories | Core Responsibilities |
| :--- | :--- | :--- |
| **Member 1 — Dataset & Data Quality** | `src/prooflens/data/`, `tests/data/` | Dataset manifest contract (`manifest.py`), directory scanner (`scanner.py`), image integrity & duplicate auditing (`validation.py`), and PyTorch `ProofLensDataset` (`dataset.py`). |
| **Member 2 — Models & Training** | `src/prooflens/models/`, `src/prooflens/training/`, `tests/models/`, `tests/training/` | Model interface contract (`factory.py` for `resnet50`, `densenet121`, `mobilenetv3_large`, `convnext_tiny`), CPU smoke training loop (`trainer.py`), and checkpoint serialization (`checkpoint.py`). |
| **Member 3 — Evaluation & Research** | `src/prooflens/evaluation/`, `src/prooflens/experiments/`, `tests/evaluation/` | Prediction output contract (`metrics.py`), subgroup metric aggregation (`subgroups.py`), paired bootstrap & Difference-in-Differences statistical utilities (`statistics.py`), and experiment config runner (`runner.py`). |
| **Member 4 — Infrastructure & Reproducibility** | `configs/`, `scripts/`, `notebooks/`, `.github/workflows/`, `pyproject.toml`, `README.md`, `CONTRIBUTING.md` | Project packaging, CLI entrypoints, synthetic fixture generation, GitHub Actions CI, and Local-to-Kaggle compatibility (`notebooks/kaggle_runner.ipynb`). |

---

## 2. Local Environment Setup

Target Python version: **Python 3.11** (CPU-only local development).

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

### Linux / macOS
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

---

## 3. Branching & Commit Conventions

### Branch Naming
Never commit directly to `main`. Create focused feature or fix branches from `main`:
- `feat/<short-description>` — New features, contracts, or experiment modules (e.g., `feat/cross-generator-eval`)
- `fix/<short-description>` — Bug fixes or schema corrections (e.g., `fix/single-class-auroc`)
- `docs/<short-description>` — Documentation or report updates (e.g., `docs/phase0b-protocol`)
- `test/<short-description>` — New unit tests or synthetic fixture improvements (e.g., `test/bootstrap-edge-cases`)

### Commit Messages
Write concise, imperative commit messages in English:
- `feat(data): add strict generator mapping validation in scanner`
- `fix(evaluation): return None instead of 0.0 for undefined single-class AUROC`
- `test(models): verify (B, 1) logit output contract across all four CNNs`

---

## 4. Pull Request Workflow & Code Review Checklist

1. **Before opening a PR**, run the full local validation suite from the repository root:
   ```bash
   python scripts/generate_fixtures.py
   ruff check .
   ruff format --check .
   python -m pytest -q
   python scripts/audit.py --root data/fixtures
   python scripts/train.py --config configs/smoke.yaml
   ```
2. **Open a Pull Request** targeting `main` and request review from the relevant module owner (at least one peer review required).
3. **Reviewer Checklist**:
   - [ ] Does the code adhere to the Manifest, Model (`(batch_size, 1)` raw logits), and Prediction contracts?
   - [ ] Are all paths handled via `pathlib.Path` without hardcoded Windows drive letters or `/kaggle/input` strings in core modules?
   - [ ] Are unit tests included and passing on CPU with `pretrained=False` and synthetic fixtures?
   - [ ] Does `ruff check .` and `ruff format --check .` pass with zero warnings?
   - [ ] Are raw images, `.pt` checkpoints, credentials (`kaggle.json`, `.env`), and temporary artifacts excluded from Git?

---

## 5. Dataset Handling & Experiment Reproducibility Rules

1. **Never download the ~8 GB Tiny GenImage dataset for local unit testing or commit raw images to Git.** Local engineering relies exclusively on deterministic synthetic fixtures (`python scripts/generate_fixtures.py`).
2. **Never guess labels or generators.** Folder-to-label mappings must be explicit (`REAL = 0`, `FAKE = 1`), and unknown/ambiguous folders must be rejected.
3. **Never fabricate or zero-fill undefined metrics.** Single-class evaluation subsets must report `None` (`NaN`) for undefined metrics such as AUROC or Balanced Accuracy.
4. **Always seed experiments deterministically** using `set_deterministic_seed(seed)` and version-controlled YAML files under `configs/`.
