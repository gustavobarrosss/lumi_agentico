import json
import random

import pytest

from harness import splits as sp
from harness.paths import CONFIGS


def test_deterministic_and_order_independent(hf_sizes):
    a = sp.make_splits(hf_sizes)
    shuffled = {cap: dict(random.Random(1).sample(list(d.items()), len(d))) for cap, d in hf_sizes.items()}
    assert sp.make_splits(shuffled) == a


def test_no_leak_between_dev_and_train_in_any_capability(hf_sizes):
    s = sp.make_splits(hf_sizes)
    sp.check_no_leak(s)  # não levanta
    dev = set(s["dev_domains"])
    for info in s["per_capability"].values():
        assert not dev & set(info["train"])


def test_dev_domain_is_dev_in_every_capability_that_has_it(hf_sizes):
    """Domínio que existe nas duas capacidades não pode ser dev numa e treino na outra."""
    s = sp.make_splits(hf_sizes)
    for dom in s["dev_domains"]:
        for cap, per_cap in hf_sizes.items():
            if dom in per_cap:
                assert dom in s["per_capability"][str(cap)]["dev"]


def test_minimum_dev_domains_per_capability(hf_sizes):
    s = sp.make_splits(hf_sizes)
    for info in s["per_capability"].values():
        assert len(info["dev"]) >= sp.MIN_DEV_PER_CAP
        assert 0.05 <= info["dev_share_of_input_bytes"] <= 0.40


def test_fails_when_capability_too_small():
    with pytest.raises(ValueError):
        sp.make_splits({1: {"a": 1, "b": 1}}, min_dev_per_cap=4)


def test_check_no_leak_detects_leak():
    bad = {"dev_domains": ["x"], "per_capability": {"1": {"dev": ["x"], "train": ["x", "y"]}}}
    with pytest.raises(AssertionError):
        sp.check_no_leak(bad)


def test_committed_splits_match_code(hf_sizes):
    """configs/splits.json (versionado, usado pelos dois times) tem que ser o que o código gera."""
    committed = json.loads((CONFIGS / "splits.json").read_text(encoding="utf-8"))
    assert committed == sp.make_splits(hf_sizes)
    sp.check_no_leak(committed)
