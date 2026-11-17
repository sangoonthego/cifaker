# ProofLens — AI-Generated Image Detection Research Codebase

**Research Title:** *Research and Development of an AI-Generated Image Detection Model*  
**Current Phase:** **Phase 0A — Local Engineering Foundation**

---

## 1. Project Overview & Research Objectives

**ProofLens** is a collaborative machine learning research codebase designed for a four-member team investigating binary classification of **REAL (`0`)** photographs versus **AI-generated (`1`, FAKE)** images on the **Tiny GenImage** dataset (`yangsangtai/tiny-genimage`, ~8 GB hosted on Kaggle).

### Core Research Objectives
- **Architecture Benchmarking:** Compare four canonical CNN families:
  1. `resnet50` (ResNet-50)
  2. `densenet121` (DenseNet-121)
  3. `mobilenetv3_large` (MobileNetV3-Large)
  4. `convnext_tiny` (ConvNeXt-Tiny)
- **Cross-Generator Generalization:** Evaluate seen vs. unseen generative models (GANs and Diffusion models) using generator-holdout protocols.
- **Robustness & Sensitivity Analysis:** Measure detection stability under compression, resizing, and noise perturbations.
- **Failure Mode & Subgroup Analysis:** Decompose errors across individual generators, resolutions, and image formats.
- **Feature Interactions & Statistical Validation:** Quantify augmentation and architectural interactions via 2×2 factorial Difference-in-Differences (DiD) contrasts and paired bootstrap confidence intervals.

---

## 2. Execution Separation: LOCAL vs. KAGGLE vs. GITHUB

To keep local development fast, reproducible, and accessible on standard CPU laptops without downloading the ~8 GB dataset locally, ProofLens strictly separates responsibilities across three environments:

| Environment | Scope & Responsibilities | Dataset Used | Compute |
| :--- | :--- | :--- | :--- |
| **LOCAL** | Package development, data/model/evaluation contract design, deterministic synthetic fixtures, CPU smoke testing, static analysis (`ruff`), and unit tests (`pytest`). | 24 deterministic 64×64 RGB synthetic fixtures (`data/fixtures/`) | CPU-only (`pretrained=False`) |
| **KAGGLE** | **Phase 0B** full dataset audit, **Phase 1+** GPU model training, cross-generator evaluation, and large-scale benchmarking via `notebooks/kaggle_runner.ipynb`. | Full `yangsangtai/tiny-genimage` mounted at `/kaggle/input/tiny-genimage` | Kaggle GPU (`cuda`) |
| **GITHUB** | Version control, 4-member branch/PR collaboration (`CONTRIBUTING.md`), and automated CPU Continuous Integration (`.github/workflows/ci.yml`). | Synthetic fixtures generated during CI run | GitHub Actions CPU runner |

> **Important Note:** Phase 0A establishes and validates the complete local engineering foundation. Phase 0B (full dataset audit on Kaggle) and Phase 1 (GPU benchmark training) have **not** been executed yet.

---

## 3. Architecture & Directory Structure

ProofLens uses a clean `src/` Python package layout with strict separation between dataset handling, model architecture, training, evaluation, and experiment configuration:

```text
prooflens/
|-- configs/
|   |-- data.yaml                         # Explicit label, split, and generator mappings
|   |-- smoke.yaml                        # Local CPU smoke experiment config
|   |-- models/
|   |   |-- architectures.yaml            # Planned CNN architecture specifications
|   |-- experiments/
|       |-- kaggle_template.yaml          # Template for future Kaggle GPU runs
|-- data/
|   |-- fixtures/                         # Ignored directory for generated synthetic RGB fixtures
|   |-- manifests/                        # Manifest CSV storage
|-- src/
|   |-- prooflens/
|       |-- __init__.py
|       |-- data/                         # [Member 1] Scanner, Manifest, Validation, PyTorch Dataset
|       |   |-- __init__.py
|       |   |-- scanner.py
|       |   |-- manifest.py
|       |   |-- validation.py
|       |   |-- dataset.py
|       |-- models/                       # [Member 2] Torchvision CNN Factory
|       |   |-- __init__.py
|       |   |-- factory.py
|       |-- training/                     # [Member 2] CPU Smoke Trainer & Checkpointing
|       |   |-- __init__.py
|       |   |-- trainer.py
|       |   |-- checkpoint.py
|       |-- evaluation/                   # [Member 3] Metrics, Subgroups & Statistical Utilities
|       |   |-- __init__.py
|       |   |-- metrics.py
|       |   |-- subgroups.py
|       |   |-- statistics.py
|       |-- experiments/                  # [Member 3 & 4] Config Validation & Smoke Runner
|           |-- __init__.py
|           |-- runner.py
|-- scripts/
|   |-- generate_fixtures.py              # Deterministic synthetic fixture generator
|   |-- audit.py                          # Dataset directory scanner & audit CLI
|   |-- train.py                          # CPU smoke training CLI
|   |-- evaluate.py                       # Prediction CSV evaluation CLI
|   |-- package.py                        # Source code packager for Kaggle upload
|-- notebooks/
|   |-- kaggle_runner.ipynb               # Thin Kaggle execution scaffold
|-- tests/
|   |-- data/test_data_pipeline.py
|   |-- models/test_model_factory.py
|   |-- training/test_smoke_training.py
|   |-- evaluation/test_evaluation_and_stats.py
|-- artifacts/                            # Ignored directory for local smoke outputs
|-- reports/                              # Research reports directory
|-- .github/workflows/ci.yml              # GitHub Actions CPU CI workflow
|-- pyproject.toml
|-- requirements.txt
|-- .gitignore
|-- CONTRIBUTING.md
|-- README.md
```

---

## 4. Environment Setup & Dependency Installation

Target Python version: **Python 3.11**.

### Windows PowerShell Setup
```powershell
# 1. Create and activate local virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip and install CPU-only PyTorch & torchvision
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install ProofLens in editable mode with development tools (pytest, ruff)
python -m pip install -e ".[dev]"
```

### Linux / macOS Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

---

## 5. Synthetic Fixture Generation & Local Validation Commands

All local engineering commands run on CPU without internet access or the full Kaggle dataset:

```powershell
# 1. Verify package installation
python -c "import prooflens; print('ProofLens import OK')"

# 2. Generate 24 deterministic synthetic RGB fixtures and data/manifests/fixtures.csv
python scripts/generate_fixtures.py

# 3. Run Ruff linter and formatting checks
ruff check .
ruff format --check .

# 4. Run the automated unit test suite
python -m pytest -q

# 5. Audit the local synthetic fixtures
python scripts/audit.py --root data/fixtures

# 6. Run the CPU smoke training & prediction pipeline
python scripts/train.py --config configs/smoke.yaml

# 7. Evaluate generated smoke predictions
python scripts/evaluate.py --predictions artifacts/local_smoke/predictions_smoke.csv
```

> **Disclaimer:** Synthetic fixtures (`data/fixtures/`) consist of deterministic geometric/noise patterns for testing pipeline mechanics. Metrics computed on fixtures must never be reported as scientific findings.

---

## 6. Core Engineering Contracts

### A. Dataset Manifest Contract (`src/prooflens/data/manifest.py`)
Every dataset manifest CSV must contain the following required columns:
- `image_id` (`str`, unique identifier)
- `relative_path` (`str`, POSIX path relative to dataset root; absolute paths and `..` are rejected)
- `generator` (`str`, source generator identity preserved for cross-generator holdout splits)
- `label` (`int`, strict binary convention: `REAL = 0`, `FAKE = 1`)
- `source_split` (`str`, preserved source split e.g. `train`, `val`, `test`)

Optional audit metadata columns: `width`, `height`, `image_format`, `file_size_bytes`, `sha256`, `decode_status`.

### B. Model Interface Contract (`src/prooflens/models/factory.py`)
All four planned CNN architectures (`resnet50`, `densenet121`, `mobilenetv3_large`, `convnext_tiny`) are instantiated through a unified factory:
```python
from prooflens.models import create_model

model = create_model(name="resnet50", pretrained=False, num_classes=1)
logits = model(images)  # shape == (batch_size, 1)
```
- Models output a **single raw logit** (`shape == (batch_size, 1)`) compatible with `torch.nn.BCEWithLogitsLoss`.
- `torch.sigmoid` is **never** applied inside `model.forward()`; probability conversion belongs strictly to the prediction/evaluation layer.

### C. Prediction & Evaluation Contract (`src/prooflens/evaluation/metrics.py`)
All model evaluation tables follow the standard schema:
- `image_id`, `model`, `generator`, `y_true` (`{0, 1}`), `y_score` (sigmoid probability in `[0, 1]`), `y_pred` (`{0, 1}`), `run_id`.
- Computed metrics: `accuracy`, `precision`, `recall`, `f1`, `balanced_accuracy`, `auroc`, `average_precision`, and `confusion_matrix` (`tn`, `fp`, `fn`, `tp`).
- **Undefined Metric Safety:** On single-class subsets (e.g., evaluating recall on a single generator's FAKE-only subset), undefined metrics such as `auroc` or `balanced_accuracy` return `None` rather than fabricated zeros.

---

## 7. Kaggle Integration Strategy & Four-Member Team Workflow

### Packaging for Kaggle
When moving from local development to Kaggle Notebooks (`notebooks/kaggle_runner.ipynb`):
1. Run `python scripts/package.py` locally to create `artifacts/prooflens_code.zip` (which bundles `src/`, `configs/`, `scripts/`, and `pyproject.toml` while excluding datasets, `.venv`, and checkpoints).
2. Upload `prooflens_code.zip` as a Kaggle Dataset or clone the GitHub repository inside the Kaggle Notebook where `yangsangtai/tiny-genimage` is mounted at `/kaggle/input/tiny-genimage`.

### Four-Member Team Workflow
See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed module ownership across Member 1 (Data), Member 2 (Models/Training), Member 3 (Evaluation/Statistics), and Member 4 (Infrastructure/CI), branch naming (`feat/*`, `fix/*`, `docs/*`, `test/*`), and Pull Request review rules.

---

## 8. Research Roadmap

- **Phase 0A (Current — COMPLETE):** Local engineering foundation, contracts, synthetic fixtures, CPU smoke testing, and GitHub CI.
- **Phase 0B (Next):** Full Tiny GenImage dataset audit, SHA-256 duplicate & cross-split leakage analysis, and format/resolution confounding report executed directly on Kaggle.
- **Phase 1:** Baseline CNN training and in-domain benchmarking on Kaggle GPUs.
- **Phase 2:** Cross-generator generalization (leave-one-generator-out and GAN-vs-Diffusion holdout protocols) and robustness evaluation.
- **Phase 3:** Factorial ablation studies, subgroup failure analysis, and paired bootstrap statistical validation.
