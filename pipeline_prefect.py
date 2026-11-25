"""Prefect pipeline for the churn prediction project."""
import argparse
import subprocess
from pathlib import Path

from prefect import flow, task, get_run_logger

VENV_DIR = Path("venv")
PYTHON_BIN = VENV_DIR / "bin" / "python"
PY = str(PYTHON_BIN)
MAIN_PY = "main.py"
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def _run(cmd):
    """Run a command and make the task fail if it returns an error."""
    logger = get_run_logger()
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.stdout:
        logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stdout)
        logger.error(result.stderr)
        raise RuntimeError("Command failed: " + " ".join(cmd))


# ---------------------------------------------------------------------------
# TASKS - project / environment
# ---------------------------------------------------------------------------
@task(name="fetch-project")
def fetch_project():
    """Get the latest version of the project from the remote git repo."""
    _run(["git", "pull", "origin", "main"])


@task(name="create-venv")
def create_venv():
    """Create the virtual environment if it does not exist."""
    if not VENV_DIR.exists():
        _run(["python3", "-m", "venv", "venv"])


@task(name="install-dependencies")
def install_dependencies():
    """Install the packages listed in requirements.txt."""
    _run([PY, "-m", "pip", "install", "-r", "requirements.txt"])


# ---------------------------------------------------------------------------
# TASKS - code quality (flake8, bandit, pytest)
# ---------------------------------------------------------------------------
@task(name="format-code")
def format_code():
    """Style/format check with flake8 (pycodestyle rules)."""
    _run([PY, "-m", "flake8", "--select=E,W", "--max-line-length=100", *CODE_FILES])


@task(name="code-quality")
def check_quality():
    """Quality check with flake8 (errors, unused imports, complexity)."""
    _run([PY, "-m", "flake8", "--select=F,C90", "--max-complexity=10", *CODE_FILES])


@task(name="code-security")
def check_security():
    """Security scan with bandit (medium severity and above)."""
    _run([PY, "-m", "bandit", "-ll", "-q", *CODE_FILES])


@task(name="unit-tests")
def run_tests():
    """Run the unit tests with pytest."""
    _run([PY, "-m", "pytest", "test_model_pipeline.py", "-q"])


# ---------------------------------------------------------------------------
# TASKS - data / model
# ---------------------------------------------------------------------------
@task(name="prepare-data")
def prepare_data():
    """Data preparation."""
    _run([PY, MAIN_PY, "--prepare"])


@task(name="train-model")
def train_model():
    """Model training."""
    _run([PY, MAIN_PY, "--train"])


@task(name="save-model")
def save_model():
    """Save the trained model."""
    _run([PY, MAIN_PY, "--save"])


@task(name="load-model")
def load_model():
    """Load the saved model."""
    _run([PY, MAIN_PY, "--load"])


@task(name="evaluate-model")
def evaluate_model():
    """Evaluate the model."""
    _run([PY, MAIN_PY, "--evaluate"])


@task(name="predict")
def predict():
    """Run predictions (main.py --test prints 5 predictions)."""
    _run([PY, MAIN_PY, "--test"])


# ---------------------------------------------------------------------------
# FLOWS
# ---------------------------------------------------------------------------
@flow(name="install", log_prints=True)
def flow_install():
    """Create the venv and install dependencies."""
    create_venv()
    install_dependencies()


@flow(name="code", log_prints=True)
def flow_code():
    """Format check, quality, security, unit tests."""
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="train", log_prints=True)
def flow_train():
    """Prepare data, train and save the model."""
    prepare_data()
    train_model()
    save_model()


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    """Load the saved model and evaluate it."""
    load_model()
    evaluate_model()


@flow(name="predict", log_prints=True)
def flow_predict():
    """Run predictions."""
    predict()


@flow(name="all", log_prints=True)
def flow_all():
    """Full pipeline: fetch, install, code checks, then data/model steps."""
    fetch_project()
    install_dependencies()
    flow_code()
    prepare_data()
    train_model()
    save_model()
    evaluate_model()


FLOWS = {
    "install": flow_install,
    "code": flow_code,
    "train": flow_train,
    "evaluate": flow_evaluate,
    "predict": flow_predict,
    "all": flow_all,
}


def main():
    """Command-line entry point: python pipeline_prefect.py --flow <name>."""
    parser = argparse.ArgumentParser(description="Prefect ML pipeline")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()


if __name__ == "__main__":
    main()
