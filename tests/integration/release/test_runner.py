import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "pytest_args",
    [
        #([]),
        (["--dataset", "acs"]),
        #(["--dataset", "cps"]),
        # (["--dataset", "acs", "--population", "USA"]),
        # (["--dataset", "acs", "--population", "USA", "--state", "RI"]),
        #(["--dataset", "wic", "--year", "2015"]),
        # (["--dataset", "wic", "--population", "USA", "--state", "RI", "--year", "2015"]),
    ],
    #ids=["1", "2", "3", "4"],
    ids = ['1'],
)
def test_release_tests(
    pytest_args: list[str], release_output_dir: Path, request: pytest.FixtureRequest
) -> None:
    os.chdir(Path(__file__).parent)  # need this to access cli options from conftest.py
    base_cmd = ["pytest", "--release", "test_release.py", f"--output-dir={release_output_dir}"]
    cmd = base_cmd + pytest_args + ["--population", "USA"]

    # log using job id
    job_id = request.node.callspec.id
    log_dir = release_output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pytest_{job_id}.o"
    with open(log_file, "w") as file:
        subprocess.run(cmd, stdout=file)


@pytest.mark.parametrize("dataset", ["acs", "cps"])
def test_slow_tests(dataset: str) -> None:
    os.chdir(Path(__file__).parent)  # need this to access cli options from conftest.py
    cmd = ["pytest", "--runslow", "test_release.py", "--dataset", dataset]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
