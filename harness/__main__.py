"""CLI do harness:  python -m harness <comando> --help"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from . import analyze as an
from . import proxy as px
from . import runner, splits as sp, stage, template_check as tc
from .paths import CACHE, CAPABILITIES, DEV_DIR, RUNS, VENDOR_VAKRA, load_env_file, official_config


def _run_dir(arg: str) -> Path:
    p = Path(arg)
    return p if p.is_dir() else RUNS / arg


def cmd_splits(a) -> None:
    sizes = sp.fetch_hf_sizes() if a.source == "hf" else sp.sizes_from_dir(Path(a.source))
    s = sp.make_splits(sizes, dev_fraction=a.dev_fraction, seed=a.seed)
    sp.check_no_leak(s)
    out = sp.save_splits(s, Path(a.out) if a.out else None)
    print(f"salvo em {out}\ndev ({len(s['dev_domains'])} domínios): {', '.join(s['dev_domains'])}")
    for cap, info in s["per_capability"].items():
        print(f"  capacidade {cap}: dev={len(info['dev'])} treino={len(info['train'])} "
              f"({info['dev_share_of_input_bytes']:.1%} dos bytes de entrada no dev)")


def cmd_stage(a) -> None:
    train_root = Path(a.train_root)
    if not train_root.is_dir():
        raise SystemExit(f"{train_root} não existe. Baixe o treino: make vakra-data")
    report = stage.stage_all(train_root, sp.load_splits(), VENDOR_VAKRA, DEV_DIR)
    for cap, r in report["copied"].items():
        print(f"capacidade {cap}: {r['samples']} amostras em {len(r['domains'])} domínios" +
              (f" · AUSENTES: {r['missing']}" if r["missing"] else ""))
    m = report["mapping"]
    print(f"mapa de universos: {m['base_uuids']} UUIDs base + {m['dev_uuids']} de dev -> {DEV_DIR / 'configs'}")
    for w in m["warnings"][:5]:
        print("  aviso:", w)
    print(f"override do compose: {report['override']}\nReinicie a capacidade 1:  cd vendor/vakra && make start-capability1")


def cmd_run(a) -> None:
    run_dir = runner.run(a.tag, a.caps, a.split, a.base_url, run_id=a.run_id, max_samples=a.max_samples,
                         domains=a.domain, restart=a.restart)
    print(f"\nresultados em {run_dir}\nPróximos passos:  python -m harness eval {run_dir.name}  e  python -m harness analyze {run_dir.name}")


def cmd_eval(a) -> None:
    runner.evaluate(_run_dir(a.run), a.caps, a.domain)


def cmd_analyze(a) -> None:
    run_dir = _run_dir(a.run)
    summary = an.analyze(run_dir, a.examples)
    o = summary["overall"]
    print(f"{o['n']} perguntas · chamada válida {an._pct(o['valid_call_rate'])} · acerto {an._pct(o['accuracy'])} "
          f"({o['scored']} pontuadas)\ncategorias: {o['categories']}\nrelatório: {run_dir / 'error_analysis.md'}")


def cmd_compare(a) -> None:
    sa, sb = (json.loads((_run_dir(r) / "analysis.json").read_text(encoding="utf-8"))["summary"] for r in (a.run_a, a.run_b))
    print(f"{'capacidade':<12}{'métrica':<16}{a.run_a[:22]:>24}{a.run_b[:22]:>24}{'Δ':>9}")
    for cap in sorted(set(sa["by_capability"]) | set(sb["by_capability"])) + ["todas"]:
        x = sa["overall"] if cap == "todas" else sa["by_capability"].get(cap, {})
        y = sb["overall"] if cap == "todas" else sb["by_capability"].get(cap, {})
        for key in ("valid_call_rate", "accuracy"):
            vx, vy = x.get(key), y.get(key)
            delta = "" if vx is None or vy is None else f"{100 * (vy - vx):+.1f}pp"
            print(f"{cap:<12}{key:<16}{an._pct(vx):>24}{an._pct(vy):>24}{delta:>9}")


def cmd_proxy(a) -> None:
    log = Path(a.log) if a.log else RUNS / a.run_id / "proxy.jsonl"
    px.serve(a.upstream, log, a.host, a.port)


def cmd_check_template(a) -> None:
    if a.template:
        path = Path(a.template)
    else:
        cfg = official_config()["lumi"]
        path = tc.fetch_template(a.hf or cfg["hf_repo"], a.revision or cfg["revision"], CACHE)
    findings, renders = tc.check(path.read_text(encoding="utf-8"))
    print(f"template: {path}")
    for f in findings:
        print(f"[{f.level}] {f.text}")
    if a.show and "thinking_off" in renders:
        print("\n" + renders["thinking_off"])
    if any(f.level == "erro" for f in findings):
        raise SystemExit(1)


def vllm_command(docker: bool = False) -> list[str]:
    """Comando que serve o Lumi com tool calling. `docker=True` usa a imagem oficial do vLLM
    (caminho para a 4090 no Windows: o vLLM não roda nativo lá, só em WSL2/Linux ou container)."""
    cfg = official_config()
    s, lumi = cfg["serving"], cfg["lumi"]
    args = [lumi["hf_repo"], "--revision", lumi["revision"],
            "--served-model-name", lumi["served_model_name"], "--port", str(s["port"]),
            "--max-model-len", str(s["max_model_len"]),
            "--default-chat-template-kwargs", json.dumps({"enable_thinking": s["enable_thinking"]}),
            "--enable-auto-tool-choice", "--tool-call-parser", s["tool_call_parser"]]
    if not docker:
        return ["vllm", "serve", *args]
    hf_cache = Path.home() / ".cache" / "huggingface"  # evita baixar os pesos a cada subida
    port = str(s["port"])
    return ["docker", "run", "--rm", "--gpus", "all", "--ipc=host", "-p", f"{port}:{port}",
            "-v", f"{hf_cache}:/root/.cache/huggingface", s["docker_image"], *args]


def cmd_serve(a) -> None:
    cmd = vllm_command(docker=a.docker)
    print(" ".join(shlex.quote(c) for c in cmd))
    if a.print_only:
        return
    if shutil.which(cmd[0]) is None:
        hint = ("O vLLM não roda nativo no Windows: use `--docker` ou rode dentro do WSL2."
                if cmd[0] == "vllm" and sys.platform == "win32" else f"`{cmd[0]}` não está instalado aqui.")
        raise SystemExit(f"{hint} Use --print para só ver o comando.")
    raise SystemExit(subprocess.call(cmd))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("splits", help="gera configs/splits.json (dev por domínio)")
    s.add_argument("--source", default="hf", help="'hf' (lista pela API) ou pasta data/train já baixada")
    s.add_argument("--seed", type=int, default=sp.DEFAULT_SEED)
    s.add_argument("--dev-fraction", type=float, default=sp.DEFAULT_DEV_FRACTION)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_splits)

    s = sub.add_parser("stage-dev", help="monta work/dev + mapa de universos + override do compose")
    s.add_argument("--train-root", default=str(VENDOR_VAKRA / "data" / "train"))
    s.set_defaults(fn=cmd_stage)

    s = sub.add_parser("run", help="roda o agente oficial da VAKRA contra o Lumi")
    s.add_argument("--tag", required=True, help="nome curto, ex.: baseline, sft-v1")
    s.add_argument("--run-id", help="use o mesmo id do `proxy` para o log ficar junto")
    s.add_argument("--split", choices=["dev", "test"], default="dev")
    s.add_argument("--caps", type=int, nargs="+", choices=list(CAPABILITIES), default=list(CAPABILITIES))
    s.add_argument("--base-url", default=os.environ.get("LUMI_BASE_URL", "http://localhost:8001/v1"),
                   help="endpoint OpenAI-compatível do Lumi (default: $LUMI_BASE_URL do .env, ou o proxy local)")
    s.add_argument("--max-samples", type=int, help="por domínio; use 1-2 para fumaça")
    s.add_argument("--domain", action="append")
    s.add_argument("--restart", action="store_true")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("eval", help="avaliador oficial (juiz LLM; precisa de API_KEY da Groq)")
    s.add_argument("run")
    s.add_argument("--caps", type=int, nargs="+", choices=list(CAPABILITIES), default=list(CAPABILITIES))
    s.add_argument("--domain", action="append")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("analyze", help="classifica falhas e gera error_analysis.md")
    s.add_argument("run")
    s.add_argument("--examples", type=int, default=3)
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("compare", help="compara duas execuções")
    s.add_argument("run_a")
    s.add_argument("run_b")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("proxy", help="proxy de gravação entre o agente e o vLLM")
    s.add_argument("--upstream", default="http://localhost:8000")
    s.add_argument("--run-id", default="proxy")
    s.add_argument("--log")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8001)
    s.set_defaults(fn=cmd_proxy)

    s = sub.add_parser("check-template", help="renderiza o chat template do Lumi e confere tool calling")
    s.add_argument("--template", help="caminho local do chat_template.jinja")
    s.add_argument("--hf")
    s.add_argument("--revision")
    s.add_argument("--show", action="store_true", help="imprime o texto renderizado")
    s.set_defaults(fn=cmd_check_template)

    s = sub.add_parser("serve", help="sobe o Lumi no vLLM com tool calling (Linux/WSL2, ou --docker)")
    s.add_argument("--docker", action="store_true", help="usa a imagem vllm/vllm-openai com --gpus all")
    s.add_argument("--print", dest="print_only", action="store_true", help="só imprime o comando")
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    load_env_file()  # antes do parser: LUMI_BASE_URL do .env vira o default de --base-url
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
