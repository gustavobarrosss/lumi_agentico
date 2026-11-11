import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def hf_sizes() -> dict[int, dict[str, int]]:
    """Listagem real de domínios de treino (capacidades 1 e 2) tirada da API do HF."""
    raw = json.loads((ROOT / "tests" / "fixtures" / "hf_sizes.json").read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}
