"""Clona a IBM/vakra no commit fixado em configs/official.json e cria o venv dela.

Roda igual no Windows e no Linux. É idempotente: pode ser executado de novo.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "vakra"


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("A VAKRA exige Python >= 3.11 (você está com %d.%d)." % sys.version_info[:2])
    cfg = json.loads((ROOT / "configs" / "official.json").read_text(encoding="utf-8"))["vakra"]

    if not (VENDOR / ".git").exists():
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", cfg["repo"], str(VENDOR)])
    sh(["git", "fetch", "--quiet", "origin"], cwd=VENDOR)
    sh(["git", "checkout", "--quiet", cfg["commit"]], cwd=VENDOR)

    venv = VENDOR / ".venv"
    if not venv.exists():
        sh([sys.executable, "-m", "venv", str(venv)])
    py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    sh([str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    sh([str(py), "-m", "pip", "install", "--quiet", "-e", ".[init]"], cwd=VENDOR)
    sh([str(py), "-m", "pip", "install", "--quiet", "-r", "requirements_benchmark.txt"], cwd=VENDOR)
    print(f"\nVAKRA pronta em {VENDOR} @ {cfg['commit'][:8]}\nPróximo passo: make vakra-data")


if __name__ == "__main__":
    main()
