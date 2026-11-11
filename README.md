# Lumi Agêntico — harness

Harness do Desafio Lumi Agêntico (PDC / Lumi, 10/11 a 30/11/2026): mede o **Lumi 4B** fazendo tool calling no benchmark **VAKRA** (capacidades 1 e 2), sempre pelo **agente e avaliador oficiais** da IBM, sem alterar o código deles e com parâmetros fixos (`configs/official.json`). Mesmo dev e mesmos parâmetros em toda execução, para o baseline e cada checkpoint serem comparáveis.

> Análise completa do desafio, achados e perguntas abertas: [docs/01-analise-do-desafio.md](docs/01-analise-do-desafio.md).

## Como as peças se encaixam

Tudo roda numa máquina só (a da 4090):

```
┌──────────────────┐   OpenAI API   ┌─────────┐        ┌──────────────────────────────┐
│ vLLM + Lumi 4B   │ <───────────── │  proxy  │ <───── │ agente oficial (LangGraph)   │
│ (make serve)     │     :8000      │ :8001   │        │ ── MCP/docker exec ──> VAKRA │
└──────────────────┘                │ grava   │        │    containers (cap. 1 e 2)   │
                                    │ tudo    │        └──────────────────────────────┘
                                    └─────────┘                     │
                                       proxy.jsonl         runs/<id>/<capacidade>/<domínio>.json
                                           └───────────────┬───────────────┘
                                              eval (juiz Groq) → analyze → error_analysis.md
```

- **vLLM:** não roda nativo no Windows. No Linux ou no WSL2 use `make serve`; no Windows com Docker Desktop (que a VAKRA já exige) use `make serve SERVE_ARGS=--docker`, que sobe a imagem `vllm/vllm-openai` com `--gpus all` (rode pelo `make` e não cole o comando impresso no PowerShell: as aspas do JSON quebram). Precisa do driver NVIDIA atual (e do NVIDIA Container Toolkit no Linux). Nenhum dos dois caminhos foi testado ainda com o NemotronH do Lumi.
- **Uma GPU só:** servir o Lumi e treinar disputam os 24 GB. Pare o `serve` antes de treinar e suba de novo para avaliar cada checkpoint.
- O agente fala com o modelo por HTTP (`--base-url`, padrão `http://localhost:8001/v1`, o proxy), então o harness não depende de como o modelo é servido.

## Primeiro baseline

Na máquina da 4090, depois do `git clone`, precisa estar instalado: **Git**, **Python 3.11+**, **make** (no Windows, rode tudo pelo Git Bash), **Docker** (Docker Desktop com WSL2 no Windows), driver NVIDIA atual, ~35 GB livres e a chave da Groq (`API_KEY`, só para o `eval`). O `HF_TOKEN` do `.env` só é necessário se o Hugging Face recusar o download do modelo ou dos dados (não testei sem ele). O repositório não traz dados, pesos nem o clone da VAKRA: tudo isso é baixado pelos alvos abaixo.

```bash
cp .env.example .env                 # preencha API_KEY (chave da Groq); o harness lê o .env sozinho
make setup && make test              # harness + 30 testes, sem GPU
make vakra-clone                     # IBM/vakra no commit fixado + venv
make vakra-data                      # ~35 GB (treino público)
make vakra-up                        # containers (Docker com 8 GB+ de RAM)
make stage                           # work/dev + mapa de universos + override; depois reinicie a cap. 1:
                                     #   cd vendor/vakra && make start-capability1

make serve                           # (terminal 1) Lumi no vLLM, tool calling, thinking desligado
make proxy RUN_ID=baseline-1         # (terminal 2) grava o que o modelo recebe e devolve
make smoke RUN_ID=baseline-1         # 1 pergunta por domínio de dev: valida o encanamento
make baseline RUN_ID=baseline-1      # dev completo
make eval RUN_ID=baseline-1          # avaliador oficial (gasta cota da Groq)
make analyze RUN_ID=baseline-1       # runs/baseline-1/error_analysis.md
```

Sem GPU nem Docker já dá para: `make test`, `make template` (confere o formato de tool call do Lumi) e `python -m harness serve --print`.

## Comandos

| comando | o que faz |
|---|---|
| `python -m harness splits` | gera `configs/splits.json` (dev por domínio, sem vazamento entre capacidades) |
| `stage-dev` | copia os domínios de dev, estende o mapa de universos da cap. 1, gera o override do compose |
| `serve [--docker] [--print]` | sobe (ou só imprime) o vLLM: `qwen3_coder`, `enable_thinking=false`, revisão fixada |
| `proxy` | proxy de gravação (JSONL + tools deduplicadas) |
| `run` | roda o `benchmark_runner.py` oficial; grava `run_meta.json` (commits, parâmetros) |
| `eval` | roda o avaliador oficial no dev |
| `analyze` | classifica cada falha, mede **chamada válida** e acerto, e junta as estatísticas do proxy |
| `compare A B` | diferença em pontos percentuais entre duas execuções |
| `check-template` | renderiza o chat template real do Lumi e confere o formato |

## Estrutura

```
configs/official.json   parâmetros e commits fixados (VAKRA, Lumi) — a confirmar com a organização
configs/splits.json     domínios de dev × treino (versionado; um teste garante que bate com o código)
harness/                splits · stage · runner · proxy · analyze · template_check · CLI
scripts/setup_vakra.py  clone fixado + venv da VAKRA (Windows e Linux)
tests/                  30 testes, sem GPU nem Docker
docs/                   análise do desafio
vendor/ work/ runs/     (ignorados pelo git) clone da VAKRA · conjunto de dev · saídas
```

## Regras que o harness protege

- **Dev só com domínios que não entram no treino.** Treine com `per_capability.<n>.train` de `configs/splits.json`, nunca com `dev`.
- **Nada do código da VAKRA é alterado**; só variáveis de ambiente e argumentos. `run` avisa se o clone sair do commit fixado.
- **Nenhuma `OPENAI_API_KEY` real é enviada ao endpoint do Lumi** (o harness usa `EMPTY`, ou `LUMI_API_KEY` se você definir).

## Estado

Testado sem GPU/Docker (ver seção 7 da análise). **Ainda não rodei o ciclo completo** (vLLM + containers + avaliador): o `make smoke` existe para expor no primeiro dia o que só aparece com o ambiente de pé. O pipeline de treino e as capacidades 3–4 ficaram de fora.
