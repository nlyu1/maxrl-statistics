import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_repo_base() -> Path:
    value = os.environ.get("REPO_BASE")
    if value is None:
        raise RuntimeError(
            "REPO_BASE is not set. "
            "Copy .env.example to .env and set REPO_BASE to the repo root."
        )
    return Path(value)


def chdir_repo_base() -> None:
    os.chdir(get_repo_base())
