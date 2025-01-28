import os
import subprocess
import time
from pathlib import Path

import pytest

from tests.integration.release.conftest import CLI_DEFAULT_OUTPUT_DIR


@pytest.mark.parametrize(
    "pytest_args",
    [
        ([]),
        (["--dataset", "acs"]),
        (["--dataset", "cps"]),
        # (["--dataset", "acs", "--population", "USA"]),
        # (["--dataset", "acs", "--population", "USA", "--state", "RI"]),
        (["--dataset", "wic", "--year", "2015"]),
        # (["--dataset", "wic", "--population", "USA", "--state", "RI", "--year", "2015"]),
    ],
    ids=["1", "2", "3", "4"],
)
def test_release_tests(pytest_args: list[str], request: pytest.FixtureRequest) -> None:
    output_dir_name = request.config.getoption("--output-dir", default=CLI_DEFAULT_OUTPUT_DIR)
    timestamped_dir = Path(output_dir_name) / f"{time.strftime('%Y%m%d_%H%M%S')}"
    timestamped_dir.mkdir(parents=True, exist_ok=False)

    os.chdir(Path(__file__).parent)  # need this to access cli options from conftest.py
    base_cmd = [
        "pytest",
        "--release",
        "test_release.py",
        "--check-max-tb=1000",
        f"--output-dir={timestamped_dir}",
    ]
    cmd = base_cmd + pytest_args

    # log using job id
    job_id = request.node.callspec.id
    log_dir = timestamped_dir.resolve() / "logs"
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
