"""Executa o agente e o avaliador OFICIAIS da VAKRA com os parâmetros de configs/official.json.

Nada do código da VAKRA é alterado: só variáveis de ambiente e argumentos de linha de comando.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .paths import CAPABILITIES, DEV_DIR, EVALUATOR_NAMES, ROOT, RUNS, VENDOR_VAKRA, official_config


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def vakra_python() -> str:
    """Python do venv da VAKRA se existir (make vakra-clone cria), senão o atual."""
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        p = VENDOR_VAKRA / rel
        if p.exists():
            return str(p)
    return sys.executable


def check_vendor(cfg: dict) -> str | None:
    """Garante o clone da VAKRA no commit fixado. Retorna o commit atual."""
    if not (VENDOR_VAKRA / "benchmark_runner.py").exists():
        raise SystemExit("VAKRA não encontrada em vendor/vakra. Rode: make vakra-clone")
    head = _git(["rev-parse", "HEAD"], VENDOR_VAKRA)
    pinned = cfg["vakra"]["commit"]
    if head != pinned:
        print(f"AVISO: vendor/vakra está em {str(head)[:8]}, o fixado é {pinned[:8]}. "
              "Resultados não serão comparáveis com os do outro time.", file=sys.stderr)
    return head


def capability_dir(cap_id: int, split: str) -> Path:
    if split == "dev":
        return DEV_DIR / CAPABILITIES[cap_id]
    if split == "test":
        return VENDOR_VAKRA / "data" / "test" / CAPABILITIES[cap_id]
    raise ValueError(f"split desconhecido: {split}")


def build_env(cfg: dict, base_url: str, split: str, caps: list[int]) -> dict[str, str]:
    env = dict(os.environ)
    # Nunca repassar uma OPENAI_API_KEY real para o endpoint local/túnel do Lumi.
    env["OPENAI_API_KEY"] = os.environ.get("LUMI_API_KEY", "EMPTY")
    env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_API_BASE"] = base_url  # nome antigo lido por versões do langchain-openai
    env["AGENT_TIMEOUT_SECONDS"] = str(cfg["agent"]["agent_timeout_seconds"])
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(VENDOR_VAKRA) + os.pathsep + env.get("PYTHONPATH", "")
    for cap_id in caps:
        env[f"CAPABILITY_{cap_id}_DIR"] = str(capability_dir(cap_id, split).resolve())
    return env


def build_run_command(cap_id: int, out_dir: Path, cfg: dict, python: str, *, max_samples: int | None = None,
                      domains: list[str] | None = None, restart: bool = False) -> list[str]:
    a = cfg["agent"]
    cmd = [python, "benchmark_runner.py", "--capability_id", str(cap_id),
           "--provider", a["provider"], "--model", cfg["lumi"]["served_model_name"],
           "--temperature", str(a["temperature"]), "--top-k-tools", str(a["top_k_tools"]),
           "--output", str(out_dir.resolve())]
    if a.get("max_iterations") is not None:
        cmd += ["--max-iterations", str(a["max_iterations"])]
    if max_samples:
        cmd += ["--max-samples-per-domain", str(max_samples)]
    for d in domains or []:
        cmd += ["--domain", d]
    if restart:
        cmd.append("--restart")
    return cmd


def new_run_id(tag: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M}-{tag}"


def run(tag: str, caps: list[int], split: str, base_url: str, *, run_id: str | None = None,
        max_samples: int | None = None, domains: list[str] | None = None, restart: bool = False) -> Path:
    cfg = official_config()
    vakra_head = check_vendor(cfg)
    run_id = run_id or new_run_id(tag)
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    python, env = vakra_python(), build_env(cfg, base_url, split, caps)

    dirty = bool(_git(["status", "--porcelain"], ROOT))
    meta = {
        "run_id": run_id, "started_at": datetime.now().isoformat(timespec="seconds"), "split": split,
        "capabilities": caps, "model": cfg["lumi"]["served_model_name"], "lumi_revision": cfg["lumi"]["revision"],
        "base_url": base_url, "agent": cfg["agent"], "max_samples_per_domain": max_samples, "domains": domains,
        "vakra_commit": vakra_head, "harness_commit": _git(["rev-parse", "HEAD"], ROOT), "harness_dirty": dirty,
        "official_config_sha1": hashlib.sha1((ROOT / "configs" / "official.json").read_bytes()).hexdigest(),
        "exit_codes": {},
    }
    for cap_id in caps:
        out_dir = run_dir / CAPABILITIES[cap_id]
        cmd = build_run_command(cap_id, out_dir, cfg, python, max_samples=max_samples, domains=domains, restart=restart)
        print(f"\n== capacidade {cap_id}: {' '.join(cmd)}")
        t0 = time.time()
        meta["exit_codes"][str(cap_id)] = subprocess.run(cmd, cwd=VENDOR_VAKRA, env=env).returncode
        meta.setdefault("seconds", {})[str(cap_id)] = round(time.time() - t0, 1)
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir


def evaluate(run_dir: Path, caps: list[int], domains: list[str] | None = None) -> list[Path]:
    """Avaliador oficial (juiz LLM). Só funciona no split dev: o GT do teste é escondido."""
    if not os.environ.get("API_KEY"):
        raise SystemExit("Defina API_KEY com a chave da Groq (juiz openai/gpt-oss-120b). Veja .env.example.")
    cfg = official_config()
    check_vendor(cfg)
    env = dict(os.environ)
    env.setdefault("JUDGE_BACKEND", cfg["judge"]["backend"])
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(VENDOR_VAKRA) + os.pathsep + env.get("PYTHONPATH", "")
    outputs = []
    for cap_id in caps:
        pred = run_dir / CAPABILITIES[cap_id]
        gt = DEV_DIR / CAPABILITIES[cap_id] / "output"
        if not pred.is_dir() or not gt.is_dir():
            raise SystemExit(f"faltam dados para avaliar a capacidade {cap_id}: {pred} / {gt}")
        out = run_dir / "eval" / f"capability_{cap_id}.results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [vakra_python(), "evaluator/evaluator.py", "--capability_name", EVALUATOR_NAMES[cap_id],
               "--gt_root", str(gt.resolve()), "--pred_root", str(pred.resolve()), "--output", str(out.resolve())]
        if domains:
            cmd += ["--domains", *domains]
        print(f"\n== avaliando capacidade {cap_id}: {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=VENDOR_VAKRA, env=env).returncode
        if rc != 0:
            raise SystemExit(f"avaliador saiu com código {rc} (capacidade {cap_id})")
        outputs.append(out)
    return outputs
