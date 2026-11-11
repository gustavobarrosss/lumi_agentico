# Harness do Desafio Lumi Agêntico. `make help` lista os alvos.
# No Windows rode pelo Git Bash. PY=python3 make ... se o seu Python se chamar python3.

PY ?= python
# Calculado uma vez por invocação do make (com ?= o date seria reavaliado a cada uso).
ifndef RUN_ID
RUN_ID := baseline-$(shell date +%Y%m%d-%H%M)
endif
TAG ?= baseline
# Se vazio, o harness usa LUMI_BASE_URL do .env (ou o proxy local em :8001).
BASE_URL ?=
CAPS ?= 1 2
BASEURL_ARG := $(if $(BASE_URL),--base-url $(BASE_URL),)
SAMPLES ?=
SERVE_ARGS ?=

# Python do venv da VAKRA (Linux/macOS: bin, Windows: Scripts); se ainda nao existir, usa $(PY).
VAKRA_VENV_PY := $(firstword $(wildcard vendor/vakra/.venv/bin/python vendor/vakra/.venv/Scripts/python.exe))
VAKRA_PY := $(if $(VAKRA_VENV_PY),$(abspath $(VAKRA_VENV_PY)),$(PY))

.PHONY: help setup test vakra-clone vakra-data vakra-up stage template serve proxy smoke baseline eval analyze

# Texto do help sem acentos de proposito: nao depende da codificacao do terminal.
help:
	@echo "Preparacao (uma vez)"
	@echo "  make setup         instala o harness (pip install -e .[dev])"
	@echo "  make test          testes do harness (sem GPU, sem Docker)"
	@echo "  make vakra-clone   clona a VAKRA no commit fixado + venv"
	@echo "  make vakra-data    baixa dados (~35 GB) com o make download da VAKRA"
	@echo "  make vakra-up      sobe os containers da VAKRA (Docker com 8 GB+ de RAM)"
	@echo "  make stage         monta work/dev (dominios de dev) + mapa de universos + override"
	@echo "Servir o Lumi (GPU)"
	@echo "  make serve                       vLLM nativo (Linux ou WSL2), porta 8000"
	@echo "  make serve SERVE_ARGS=--docker   vLLM em container (4090 no Windows)"
	@echo "  make template      confere o chat template do Lumi (nao precisa de GPU)"
	@echo "Rodar e medir"
	@echo "  make proxy RUN_ID=x     proxy de gravacao (porta 8001 -> 8000)"
	@echo "  make smoke RUN_ID=x     1 pergunta por dominio de dev, para validar o encanamento"
	@echo "  make baseline RUN_ID=x  conjunto de dev completo"
	@echo "  make eval RUN_ID=x      avaliador oficial (precisa de API_KEY da Groq)"
	@echo "  make analyze RUN_ID=x   relatorio de erros em runs/x/error_analysis.md"

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

vakra-clone:
	$(PY) scripts/setup_vakra.py

# O make download da VAKRA só baixa o treino quando NÃO há acesso ao teste gated.
vakra-data:
	cd vendor/vakra && "$(VAKRA_PY)" benchmark_setup.py --download-data

vakra-up:
	cd vendor/vakra && docker compose up -d

stage:
	$(PY) -m harness stage-dev

template:
	$(PY) -m harness check-template

serve:
	$(PY) -m harness serve $(SERVE_ARGS)

proxy:
	$(PY) -m harness proxy --run-id $(RUN_ID)

smoke:
	$(PY) -m harness run --tag smoke --run-id $(RUN_ID) --caps $(CAPS) $(BASEURL_ARG) --max-samples 1

baseline:
	$(PY) -m harness run --tag $(TAG) --run-id $(RUN_ID) --caps $(CAPS) $(BASEURL_ARG) $(if $(SAMPLES),--max-samples $(SAMPLES),)

eval:
	$(PY) -m harness eval $(RUN_ID) --caps $(CAPS)

analyze:
	$(PY) -m harness analyze $(RUN_ID)
