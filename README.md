# MLOps Iris CI Quality Gates

This small university project demonstrates Machine Learning Continuous
Integration for the Kaggle Iris Species dataset. It validates data, prepares
features, trains a reproducible scikit-learn classifier, evaluates it, and
runs quality gates with pytest and Ruff. The scope ends at CI: there is no
model serving, deployment, Docker, Kubernetes, cloud infrastructure, or CD.
This repository is intentionally kept small for classroom CI practice.

## Dataset

The checked-in file is `data/raw/Iris.csv`, copied unchanged from the supplied
Kaggle download. The observed schema has 150 rows and these columns:

| Column | Role |
| --- | --- |
| `Id` | Identifier, excluded from model inputs |
| `SepalLengthCm` | Numeric feature |
| `SepalWidthCm` | Numeric feature |
| `PetalLengthCm` | Numeric feature |
| `PetalWidthCm` | Numeric feature |
| `Species` | Target with three Iris classes |

The dataset source is the Kaggle **Iris Species** listing, originally based on
the UCI Iris dataset. Dataset terms and attribution are separate from this
repository's MIT license. Review the applicable Kaggle listing and original
UCI terms before redistributing the data outside this assignment.

## Project structure

```text
.
├── .github/workflows/ci.yml
├── data/raw/Iris.csv
├── src/
│   ├── data_validation.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   ├── quality_gate.py
│   └── train.py
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Local setup on Debian 12

The project uses the local `.venv` environment and does not install packages
globally. Python 3.11 is sufficient for the local commands below; GitHub
Actions runs the same checks on Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Activate the environment again in a new terminal with:

```bash
source .venv/bin/activate
```

## Run locally

Run the quality checks and tests:

```bash
python -m compileall -q src tests
ruff check .
ruff format --check .
pytest -v --tb=short
```

Run the complete local workflow against the supplied dataset:

```bash
python -m src.train data/raw/Iris.csv
```

The command validates the CSV, excludes `Id`, creates a reproducible stratified
split, fits scaling only as part of the training pipeline, and prints training
metadata and accuracy, macro precision, macro recall, and macro F1.

Run the ML quality gate and generate its JSON report:

```bash
python -m src.quality_gate data/raw/Iris.csv --report reports/quality_report.json
```

The gate validates the real CSV before training, evaluates held-out
predictions, and requires test accuracy of at least `0.90` and macro F1 of at
least `0.90`. It also requires finite metrics in the range `[0, 1]` and valid
Iris prediction labels. The thresholds and split settings are centralized in
`src/quality_gate.py`. A passing command exits with status `0`; any failed
mandatory criterion exits with status `1`.

The report includes the dataset identifier, feature names, split sizes, random
seed, model parameters, scalar metrics, confusion matrix, per-class metrics,
threshold results, and overall `PASS` or `FAIL`. Reports are generated locally
under `reports/` but are ignored by Git.

## Continuous Integration

`.github/workflows/ci.yml` runs for pushes to `main` and pull requests
targeting `main`. It has three jobs:

1. **Code quality** installs the development requirements, compiles the Python
   files, runs `ruff check .`, and checks formatting with `ruff format --check .`.
2. **Automated tests** installs the same requirements and runs
   `pytest -v --tb=short`.
3. **ML quality gate** runs only after the first two jobs pass. It validates,
   trains, and evaluates the actual `data/raw/Iris.csv`, enforces the accuracy
   and macro-F1 thresholds, and uploads `reports/quality_report.json` as the
   `ml-quality-report` artifact. The artifact upload runs even when a generated
   report accompanies a failed gate.

The workflow requests only `contents: read` and does not use secrets or
deployment credentials. The actions use maintained major versions. For
stronger supply-chain controls, pin each action such as `actions/checkout` and
`actions/setup-python` to a verified full commit SHA, then review those pins
when upgrading.

GitHub branch protection is not enabled by this file. To require successful CI
before merging, open repository **Settings > Branches > Branch protection
rules** (or the rulesets page), add a rule for `main`, require a pull request,
and require these exact status checks: `Code quality`, `Automated tests`, and
`ML quality gate`. Push the workflow and let it run successfully at least once
so GitHub can offer the checks in the ruleset. Optionally require up-to-date
branches and an approving review. Test the rule with a pull request that
deliberately fails a check before relying on it.

When a check fails, open the workflow run and inspect the failed job's log.
Fix the reported code, formatting, data, or test issue locally, rerun the same
commands, and push a new commit. With branch protection configured, GitHub
will block merging until the required checks pass.

## Limitations and future work

This is a teaching example using one small static dataset and one logistic
regression model. It does not track experiments, version data externally,
monitor drift, persist model artifacts, serve predictions, or deploy to an
environment. Future coursework could add data versioning, model comparison,
artifact storage, and a separate CD project after these CI gates are stable.