"""Análise de erros de uma execução (Entrega 1: baseline + análise de erros).

Lê o que o runner oficial da VAKRA grava (`<domínio>.json`, `<domínio>_tools.json`), o
resultado do avaliador oficial (opcional) e o log do proxy (opcional) e classifica cada
pergunta em UMA categoria de falha. A ordem das regras importa: a primeira que casa vence.

"Chamada válida" = a pergunta teve ao menos uma tool call cujo nome existe no conjunto de
ferramentas do domínio. É a métrica do plano ("hoje o Lumi não faz nenhuma chamada válida").
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .paths import CAPABILITIES

CORRECT_THRESHOLD = 0.999
TOOL_MARKUP = re.compile(r"<tool_call>|<function=|</function>")
ERROR_RULES: list[tuple[str, re.Pattern]] = [
    ("error_timeout", re.compile(r"timed out|timeout", re.I)),
    ("error_context_overflow", re.compile(r"context length|context_length|maximum context|too many tokens|reduce the length", re.I)),
    ("error_universe_switch", re.compile(r"Universe switch failed", re.I)),
    ("error_connection", re.compile(r"Connection|ConnectError|APIConnectionError|RemoteProtocolError", re.I)),
]

CATEGORY_HELP = {
    "correct": "avaliador deu nota máxima",
    "wrong_answer": "chamou ferramentas válidas mas o avaliador deu nota < 1 (resposta errada, tool errada ou não ancorada)",
    "unscored": "executou sem erro, mas não há resultado do avaliador (rode `eval`)",
    "empty_answer": "chamou ferramentas válidas e terminou sem texto de resposta",
    "unknown_tool": "chamou ferramenta que não existe no domínio (nome alucinado)",
    "no_tool_call": "terminou sem nenhuma tool call",
    "no_tool_call_markup": "terminou sem tool call, mas o texto tem marcação <tool_call>/<function=: formato certo que o parser não leu",
    "error_timeout": "estourou o AGENT_TIMEOUT_SECONDS (loop de chamadas ou modelo lento)",
    "error_context_overflow": "prompt maior que o contexto servido (ex.: 128 ferramentas no prompt)",
    "error_universe_switch": "capacidade 1: UUID ausente do mapa de universos (veja `stage-dev`)",
    "error_connection": "agente não alcançou o modelo/ambiente",
    "error_other": "outra exceção do agente",
}


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def _calls(rec: dict) -> list[dict]:
    out = rec.get("output") or []
    if not out:
        return []
    seq = out[0].get("sequence", {})
    if isinstance(seq, list):
        calls = []
        for item in seq:
            calls.extend(item.get("tool_call", []))
        return calls
    return list(seq.get("tool_call", []))


def _answer(rec: dict) -> str:
    out = rec.get("output") or []
    ans = out[0].get("answer", "") if out else ""
    return ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False)


def classify(rec: dict, all_tools: list[str] | None, score: float | None) -> tuple[str, dict]:
    """Retorna (categoria, extras)."""
    calls = _calls(rec)
    names = [c.get("name", "") for c in calls]
    known = set(all_tools) if all_tools else None
    valid = [n for n in names if n and (known is None or n in known)]
    extras = {"n_calls": len(calls), "tools_called": names, "valid_call": bool(valid)}

    if rec.get("status") == "error":
        err = rec.get("error", "") or ""
        for cat, pat in ERROR_RULES:
            if pat.search(err):
                return cat, extras
        return "error_other", extras

    if not calls:
        return ("no_tool_call_markup" if TOOL_MARKUP.search(_answer(rec)) else "no_tool_call"), extras
    if not valid:
        return "unknown_tool", extras
    if not _answer(rec).strip():
        return "empty_answer", extras
    if score is None:
        return "unscored", extras
    return ("correct" if score >= CORRECT_THRESHOLD else "wrong_answer"), extras


def load_scores(eval_file: Path) -> dict[str, float]:
    """uuid -> nota, a partir do results.json do avaliador oficial."""
    if not eval_file.exists():
        return {}
    data = json.loads(eval_file.read_text(encoding="utf-8"))
    scores = {}
    for dom in (data.get("domains") or {}).values():
        for d in dom.get("dialogues", []):
            scores[d["uuid"]] = float(d["score"])
    return scores


def load_run(run_dir: Path) -> list[dict]:
    """Uma linha por pergunta, já classificada."""
    rows: list[dict] = []
    for cap_id, cap in CAPABILITIES.items():
        cap_dir = run_dir / cap
        if not cap_dir.is_dir():
            continue
        scores = load_scores(run_dir / "eval" / f"capability_{cap_id}.results.json")
        for f in sorted(cap_dir.glob("*.json")):
            if f.stem.endswith("_tools"):
                continue
            tools_file = cap_dir / f"{f.stem}_tools.json"
            tools_by_uuid = {}
            if tools_file.exists():
                tools_by_uuid = {t["uuid"]: t for t in json.loads(tools_file.read_text(encoding="utf-8"))}
            for rec in json.loads(f.read_text(encoding="utf-8")):
                t = tools_by_uuid.get(rec["uuid"], {})
                cat, extras = classify(rec, t.get("all_tools") or None, scores.get(rec["uuid"]))
                out0 = (rec.get("output") or [{}])[0]
                rows.append({
                    "capability": cap_id, "domain": rec.get("domain", f.stem), "uuid": rec["uuid"],
                    "query": out0.get("query", ""), "status": rec.get("status"),
                    "error": rec.get("error", ""), "answer": _answer(rec),
                    "duration_s": rec.get("duration_s", 0.0), "score": scores.get(rec["uuid"]),
                    "category": cat, **extras,
                })
    return rows


def proxy_stats(log_path: Path) -> dict:
    """Estatísticas do que o modelo realmente recebeu/devolveu (log do proxy)."""
    if not log_path.exists():
        return {}
    n = 0
    status = Counter()
    finish = Counter()
    prompt_tok, compl_tok, n_tools = [], [], []
    with_tool_calls = markup_no_calls = ctx_errors = 0
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n += 1
            status[str(rec.get("status"))] += 1
            req, resp = rec.get("request") or {}, rec.get("response") or {}
            n_tools.append(req.get("n_tools", 0))
            if rec.get("status") != 200:
                msg = json.dumps(resp, ensure_ascii=False)
                if re.search(r"context length|maximum context|too many tokens", msg, re.I):
                    ctx_errors += 1
                continue
            usage = resp.get("usage") or {}
            if "prompt_tokens" in usage:
                prompt_tok.append(usage["prompt_tokens"])
            if "completion_tokens" in usage:
                compl_tok.append(usage["completion_tokens"])
            choice = (resp.get("choices") or [{}])[0]
            finish[str(choice.get("finish_reason"))] += 1
            msg = choice.get("message") or {}
            if msg.get("tool_calls"):
                with_tool_calls += 1
            elif TOOL_MARKUP.search(msg.get("content") or ""):
                markup_no_calls += 1
    return {
        "requests": n, "http_status": dict(status), "finish_reason": dict(finish),
        "responses_with_tool_calls": with_tool_calls,
        "responses_with_markup_but_no_tool_calls": markup_no_calls,
        "context_overflow_errors": ctx_errors,
        "prompt_tokens": {"p50": _quantile(prompt_tok, .5), "p90": _quantile(prompt_tok, .9), "max": max(prompt_tok, default=None)},
        "prompt_over_8192": sum(1 for p in prompt_tok if p > 8192),
        "completion_tokens": {"p50": _quantile(compl_tok, .5), "p90": _quantile(compl_tok, .9), "max": max(compl_tok, default=None)},
        "tools_in_prompt": {"p50": _quantile(n_tools, .5), "max": max(n_tools, default=None)},
    }


def summarize(rows: list[dict]) -> dict:
    out: dict = {"overall": {}, "by_capability": {}, "by_domain": {}}

    def agg(sub: list[dict]) -> dict:
        n = len(sub)
        scored = [r for r in sub if r["score"] is not None]
        return {
            "n": n,
            "valid_call_rate": round(sum(r["valid_call"] for r in sub) / n, 4) if n else None,
            # Denominador = TODAS as perguntas: sem nota (erro do agente, ausente no avaliador) conta como erro,
            # que é como a taxa de acerto do conjunto escondido é medida. None só se o avaliador não rodou.
            "accuracy": round(sum(r["score"] >= CORRECT_THRESHOLD for r in scored) / n, 4) if scored else None,
            "scored": len(scored),
            "mean_tool_calls": round(sum(r["n_calls"] for r in sub) / n, 2) if n else None,
            "mean_duration_s": round(sum(r["duration_s"] for r in sub) / n, 2) if n else None,
            "categories": dict(Counter(r["category"] for r in sub).most_common()),
        }

    out["overall"] = agg(rows)
    by_cap: dict[int, list[dict]] = defaultdict(list)
    by_dom: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cap[r["capability"]].append(r)
        by_dom[(r["capability"], r["domain"])].append(r)
    out["by_capability"] = {str(c): agg(v) for c, v in sorted(by_cap.items())}
    out["by_domain"] = {f"{c}/{d}": agg(v) for (c, d), v in sorted(by_dom.items())}
    return out


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _clip(text: str, n: int = 110) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def render_markdown(run_dir: Path, rows: list[dict], summary: dict, proxy: dict, n_examples: int = 3) -> str:
    L: list[str] = [f"# Análise de erros — `{run_dir.name}`", ""]
    meta = run_dir / "run_meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        L += [f"- modelo: `{m.get('model')}` · split: `{m.get('split')}` · commit VAKRA: `{str(m.get('vakra_commit'))[:8]}` · harness: `{str(m.get('harness_commit'))[:8]}`", ""]

    L += ["## Resumo por capacidade", "",
          "| capacidade | perguntas | chamada válida | acerto (sobre todas) | pontuadas | tool calls/pergunta | s/pergunta |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for cap, s in summary["by_capability"].items():
        L.append(f"| {cap} | {s['n']} | {_pct(s['valid_call_rate'])} | {_pct(s['accuracy'])} | {s['scored']} | {s['mean_tool_calls']} | {s['mean_duration_s']} |")
    o = summary["overall"]
    L.append(f"| **todas** | {o['n']} | {_pct(o['valid_call_rate'])} | {_pct(o['accuracy'])} | {o['scored']} | {o['mean_tool_calls']} | {o['mean_duration_s']} |")

    L += ["", "## Categorias de falha", "", "| categoria | n | % | o que significa |", "|---|---:|---:|---|"]
    total = max(1, o["n"])
    for cat, n in o["categories"].items():
        L.append(f"| `{cat}` | {n} | {_pct(n / total)} | {CATEGORY_HELP.get(cat, '')} |")

    if proxy:
        pt, tp = proxy["prompt_tokens"], proxy["tools_in_prompt"]
        L += ["", "## O que o modelo recebeu e devolveu (proxy)", "",
              f"- requisições: {proxy['requests']} · HTTP: {proxy['http_status']} · finish_reason: {proxy['finish_reason']}",
              f"- respostas com tool_calls: {proxy['responses_with_tool_calls']} · com marcação de tool call mas sem tool_calls (parser não leu): {proxy['responses_with_markup_but_no_tool_calls']}",
              f"- tokens do prompt: p50 {pt['p50']} · p90 {pt['p90']} · máx {pt['max']} · acima de 8192: {proxy['prompt_over_8192']} · erros de estouro de contexto: {proxy['context_overflow_errors']}",
              f"- ferramentas no prompt: p50 {tp['p50']} · máx {tp['max']}",
              f"- tokens gerados: p50 {proxy['completion_tokens']['p50']} · p90 {proxy['completion_tokens']['p90']}"]

    L += ["", "## Por domínio", "", "| capacidade/domínio | n | chamada válida | acerto |", "|---|---:|---:|---:|"]
    for key, s in summary["by_domain"].items():
        L.append(f"| {key} | {s['n']} | {_pct(s['valid_call_rate'])} | {_pct(s['accuracy'])} |")

    L += ["", "## Exemplos por categoria (para ler à mão)", ""]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["category"] != "correct":
            by_cat[r["category"]].append(r)
    for cat, items in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        L += [f"### `{cat}` ({len(items)})", ""]
        for r in items[:n_examples]:
            detail = _clip(r["error"]) if r["status"] == "error" else _clip(r["answer"])
            L.append(f"- cap {r['capability']} · {r['domain']} · `{r['uuid'][:13]}` — “{_clip(r['query'])}” → tools: {r['tools_called'] or '∅'} · {detail or '(sem texto)'}")
        L.append("")
    return "\n".join(L)


def analyze(run_dir: Path, n_examples: int = 3) -> dict:
    rows = load_run(run_dir)
    if not rows:
        raise SystemExit(f"nenhum resultado em {run_dir} (esperado: <capacidade>/<domínio>.json)")
    summary = summarize(rows)
    proxy = proxy_stats(run_dir / "proxy.jsonl")
    (run_dir / "analysis.json").write_text(
        json.dumps({"summary": summary, "proxy": proxy, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "error_analysis.md").write_text(render_markdown(run_dir, rows, summary, proxy, n_examples), encoding="utf-8")
    return summary
