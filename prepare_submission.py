from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "submission_bundle"

FILES = [
    "README.md",
    "Dockerfile",
    "openenv.yaml",
    "pyproject.toml",
    "requirements.txt",
    "app.py",
    "inference.py",
    "validate_local.py",
]

DIRECTORIES = [
    "server",
    "scripts",
    "support_triage_env",
]


def main() -> None:
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True, exist_ok=True)

    for file_name in FILES:
        source = ROOT / file_name
        if source.exists():
            shutil.copy2(source, TARGET / file_name)

    for directory_name in DIRECTORIES:
        source = ROOT / directory_name
        if source.exists():
            shutil.copytree(source, TARGET / directory_name, dirs_exist_ok=True)

    print(f"Prepared submission bundle at {TARGET}")


if __name__ == "__main__":
    main()
