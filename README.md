# MLOps Iris CI Quality Gates

This project demonstrates end-to-end MLOps CI and continuous delivery for the
Kaggle Iris Species dataset. It validates data, trains and evaluates a
reproducible scikit-learn pipeline, packages the passing model, serves it with
Flask and Gunicorn, smoke-tests a Docker image, and publishes that image to
Docker Hub from successful pushes to `main`.

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
├── Dockerfile
├── .dockerignore
├── artifacts/                 # generated locally/inside CI, ignored by Git
├── data/raw/Iris.csv
├── scripts/smoke_test.py
├── src/
│   ├── api.py
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
globally. Python 3.12 is the supported runtime for the container and GitHub
Actions.

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
under `reports/` but are ignored by Git. A passing gate additionally creates
`artifacts/iris_model.joblib` (the exact evaluated Pipeline) and
`artifacts/model_metadata.json` with feature order, dataset checksum, metrics,
thresholds, dependency versions, commit SHA, and UTC build time. Artifacts are
created only after every mandatory criterion passes.

## Flask API

The API loads the packaged model once at application startup and never trains
on import or request. Run it locally with:

```bash
gunicorn --bind 0.0.0.0:8000 src.api:app
```

Endpoints:

```bash
curl http://localhost:8000/health
# {"status":"healthy"}

curl http://localhost:8000/model-info

curl -X POST http://localhost:8000/predict \
   -H 'Content-Type: application/json' \
   -d '{"SepalLengthCm":5.1,"SepalWidthCm":3.5,"PetalLengthCm":1.4,"PetalWidthCm":0.2}'
# {"prediction":"Iris-setosa"}
```

Invalid JSON, missing features, non-numeric values, non-finite values, and
unsupported predictions receive an error response. `/health` returns `503`
when the required artifacts cannot be loaded.

## Docker and smoke tests

Generate the validated artifacts, build the image, and test the running
container locally:

```bash
python -m src.quality_gate data/raw/Iris.csv \
   --report reports/quality_report.json --artifacts-dir artifacts
docker build -t iris-mlops:local .
python scripts/smoke_test.py iris-mlops:local
```

The Dockerfile uses Python 3.12 slim, installs only runtime dependencies,
runs Gunicorn on port `8000`, and uses a non-root user. It fails during the
build if the validated model files are absent. The smoke test uses bounded
readiness retries and always removes the container.

## Continuous Integration and Delivery

`.github/workflows/ci.yml` runs for pushes to `main` and pull requests
targeting `main`. It retains these jobs:

1. **Code quality** installs the development requirements, compiles the Python
   files, runs `ruff check .`, and checks formatting with `ruff format --check .`.
2. **Automated tests** installs the same requirements and runs
   `pytest -v --tb=short`.
3. **ML quality gate** runs only after the first two jobs pass. It validates,
   trains, evaluates, and packages the actual `data/raw/Iris.csv`, enforces
   the accuracy and macro-F1 thresholds, and uploads the model, metadata, and
   report as workflow artifacts. The report upload runs even when a generated
   report accompanies a failed gate.
4. **Build, smoke-test, and publish image** requires all three jobs. It
   downloads and verifies the current run's model, metadata, and `PASS`
   report, builds the image, and runs the container smoke test. Pull requests
   stop there. Only a successful push to `main` logs in and publishes both
   `sha-<FULL_COMMIT_SHA>` and `latest` tags.

The workflow requests only `contents: read` and does not publish from pull
requests or failed checks. Configure these repository secrets at **Settings >
Secrets and variables > Actions**:

- `DOCKERHUB_USERNAME`: the Docker Hub account name.
- `DOCKERHUB_TOKEN`: a Docker Hub access token with only the permission needed
   to push the `iris-mlops` repository.

The workflow uses maintained action releases. For a stricter production
supply-chain policy, replace each action version with a verified full commit
SHA and review those pins when upgrading.

GitHub Actions artifacts transfer the validated model and report between jobs;
they are not Docker Hub images. The image contains the model and inference
service. Artifact retention is seven days.

To pull and run a published image:

```bash
docker pull YOUR_DOCKERHUB_USERNAME/iris-mlops:sha-FULL_COMMIT_SHA
docker run --rm -p 8000:8000 YOUR_DOCKERHUB_USERNAME/iris-mlops:sha-FULL_COMMIT_SHA
```

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

## Preparing for AWS EC2

This implementation publishes a container image to Docker Hub but does not
deploy it to a running server. For a later EC2 deployment, provision an
instance with Docker, restrict inbound access to the required port, authenticate
to Docker Hub using a deployment-only token, pull an immutable `sha-...` tag,
run it behind a reverse proxy or load balancer, and add health checks,
monitoring, TLS, and a rollback procedure. Keep deployment credentials outside
the image and Git repository.

## Limitations and future work

This is a teaching example using one small static dataset and one logistic
regression model. It does not track experiments, version data externally,
monitor drift, or deploy to an environment. Future coursework could add data
versioning, model comparison, a private artifact registry, and an automated
EC2 deployment after these CI gates are stable.