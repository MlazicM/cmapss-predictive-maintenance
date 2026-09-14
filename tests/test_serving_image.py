"""Checks on what the Docker image is asked to contain.

The image installs ``requirements-serve.txt``, not ``requirements.txt``, and
copies only ``src/`` and ``api/``. Both are claims about the code that are easy
to break by adding one import: a new top-level ``import xgboost`` in
``src/models.py`` or a plotting helper reached from ``api/`` turns a working
Dockerfile into an ``ImportError`` at container start, and nothing in the test
suite would have noticed.

These tests check the claims directly. They do not need Docker installed, which
matters because CI and most development machines do not have it.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Installed for the notebooks and the experiments, absent from the serving image.
NOT_IN_THE_IMAGE = ["xgboost", "matplotlib", "seaborn", "jupyter", "IPython", "notebook"]


def _requirement_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        names.add(re.split(r"[<>=!\[]", line)[0].strip().lower())
    return names


@pytest.fixture(scope="module")
def serve_requirements():
    return _requirement_names(PROJECT_ROOT / "requirements-serve.txt")


def test_serving_requirements_exclude_the_notebook_stack(serve_requirements):
    assert not serve_requirements & {"jupyter", "matplotlib", "seaborn", "xgboost"}


def test_serving_requirements_keep_what_unpickling_a_scaler_needs(serve_requirements):
    """The scaler is a joblib-pickled scikit-learn object.

    Dropping scikit-learn because "the API only runs a Keras model" produces an
    image that starts, serves /health, and fails on the first /predict.
    """
    assert {"scikit-learn", "joblib", "tensorflow", "numpy", "pandas"} <= serve_requirements
    assert {"fastapi", "uvicorn", "pydantic"} <= serve_requirements


def test_importing_the_api_does_not_pull_in_the_notebook_stack():
    """Run in a subprocess: this session has already imported half of these."""
    probe = (
        "import sys; import api.main; "
        f"found = [m for m in {NOT_IN_THE_IMAGE!r} if m in sys.modules]; "
        "print(','.join(found))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    leaked = result.stdout.strip()
    assert not leaked, f"api.main imports {leaked}, which the serving image does not install"


def test_the_image_copies_every_package_the_api_imports():
    """``src`` and ``api`` are the only first-party packages the Dockerfile copies."""
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    copied = set(re.findall(r"^COPY[^\n]*?\s(\w+)/\s+\./\w+/", dockerfile, flags=re.MULTILINE))
    assert copied == {"src", "api"}


def test_artifacts_are_mounted_rather_than_baked_into_the_image():
    """A model in the image means a rebuild to ship a retrained one."""
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "models/" in dockerignore
    assert "*.keras" in dockerignore

    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "./models:/app/models:ro" in compose, "artifacts are read-only at runtime"
