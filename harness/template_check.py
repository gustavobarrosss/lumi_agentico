"""Renderiza o chat template do Lumi localmente e confere o que importa para tool calling.

Serve para (a) saber, sem GPU, exatamente que texto o modelo recebe/produz com tools e
(b) garantir que os dados de SFT sigam o MESMO formato do inferência. O ambiente do Jinja
imita o do transformers (sandbox, loopcontrols, tojson sem ASCII-escape).
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .paths import CACHE

HF_RAW = "https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def fetch_template(repo: str, revision: str = "main", cache_dir: Path = CACHE) -> Path:
    """Baixa chat_template.jinja do HF (cache local por revisão)."""
    dest = cache_dir / repo.replace("/", "__") / revision / "chat_template.jinja"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = HF_RAW.format(repo=repo, revision=revision, filename="chat_template.jinja")
        with urllib.request.urlopen(url, timeout=30) as resp:
            dest.write_bytes(resp.read())
    return dest


def _tojson(value, ensure_ascii=False, indent=None, sort_keys=False, separators=None):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys, separators=separators)


def _raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def render(template_text: str, messages: list[dict], tools: list[dict] | None = None,
           add_generation_prompt: bool = True, **template_kwargs) -> str:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    tpl = env.from_string(template_text)
    return tpl.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **template_kwargs)


# Cenário de teste: pergunta -> tool call -> resultado -> pergunta nova (cobre a regra de
# "truncar histórico de pensamento" e o formato de argumentos escalares e estruturados).
SAMPLE_TOOLS = [
    {"type": "function", "function": {
        "name": "select_data_equal_to",
        "description": "Filtra linhas onde a coluna é igual ao valor.",
        "parameters": {"type": "object", "required": ["data_label", "key_name", "value"], "properties": {
            "data_label": {"type": "string", "description": "handle do dado"},
            "key_name": {"type": "string"},
            "value": {"type": "string"},
            "extra": {"type": "array", "items": {"type": "integer"}},
        }},
    }},
]
SAMPLE_MESSAGES = [
    {"role": "system", "content": "Você é um assistente de dados."},
    {"role": "user", "content": "Quantas cervejarias há na América do Norte?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "id": "call_1", "function": {
            "name": "select_data_equal_to",
            "arguments": {"data_label": "tbl_1", "key_name": "Country", "value": "United States", "extra": [1, 2]}}},
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"handle\": \"filtered_1\", \"num_records\": 23}"},
]


@dataclass
class Finding:
    level: str  # "ok" | "aviso" | "erro"
    text: str


def check(template_text: str) -> tuple[list[Finding], dict[str, str]]:
    """Roda o cenário padrão e devolve (achados, renders)."""
    out: list[Finding] = []
    renders: dict[str, str] = {}

    def add(level: str, text: str) -> None:
        out.append(Finding(level, text))

    try:
        renders["thinking_off"] = render(template_text, SAMPLE_MESSAGES, SAMPLE_TOOLS, enable_thinking=False)
        renders["thinking_default"] = render(template_text, SAMPLE_MESSAGES, SAMPLE_TOOLS)
    except jinja2.exceptions.TemplateError as exc:
        add("erro", f"template não renderiza com tools/tool_calls (arguments como dict): {exc}")
        return out, renders

    off, default = renders["thinking_off"], renders["thinking_default"]

    if "<function>\n<name>select_data_equal_to</name>" in off:
        add("ok", "ferramentas entram no system prompt em XML (<tools><function><name>…).")
    else:
        add("erro", "definição das ferramentas não apareceu no formato esperado.")

    call = "<tool_call>\n<function=select_data_equal_to>\n<parameter=data_label>\ntbl_1\n</parameter>"
    if call in off:
        add("ok", "tool call do assistente é renderizada como <tool_call><function=…><parameter=…> "
                  "(formato qwen3_coder; no vLLM: --tool-call-parser qwen3_coder).")
    else:
        add("erro", "tool call do assistente não saiu no formato <tool_call><function=…>.")

    if "<parameter=extra>\n[1, 2]\n</parameter>" in off:
        add("ok", "argumento estruturado (lista) é serializado como JSON dentro de <parameter>.")
    else:
        add("aviso", "argumento estruturado não saiu como JSON esperado.")

    if "<|im_start|>user\n<tool_response>\n" in off:
        add("ok", "resultado da ferramenta volta como turno 'user' com <tool_response>.")
    else:
        add("erro", "resultado da ferramenta não foi embrulhado em <tool_response>.")

    if off.rstrip().endswith("<|im_start|>assistant\n<think></think>"):
        add("ok", "com enable_thinking=False o prompt termina em '<think></think>' (sem raciocínio).")
    else:
        add("erro", "com enable_thinking=False o prompt não termina em '<think></think>'.")

    if default.rstrip().endswith("<|im_start|>assistant\n<think>"):
        add("aviso", "SEM passar enable_thinking o template assume True e abre '<think>' — o cartão do "
                     "modelo exige thinking DESLIGADO e o agente oficial da VAKRA não passa esse parâmetro. "
                     "Servir com --default-chat-template-kwargs '{\"enable_thinking\": false}' e/ou "
                     "entregar o modelo com o default do chat_template.jinja trocado para False.")
    return out, renders
