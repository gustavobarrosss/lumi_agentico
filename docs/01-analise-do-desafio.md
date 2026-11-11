# Desafio Lumi Agêntico — análise

Base: `desafio-lumi-agentico-plano.pdf` (PDC / Lumi, 10/11 a 30/11/2026) + leitura do código da VAKRA (IBM/vakra @ `00dc68f`), do cartão do Lumi 4B (`ceia-br/lumi-4b` @ `27b30f4`) e do cartão do dataset. Onde digo **[lido]**, conferi no código ou no dado; **[a confirmar]** só se resolve com GPU, Docker e os ~35 GB de dados.

## 1. O que o plano pede

- Dois times (A e B, 3–4 pessoas; este harness é do nosso time) treinam o **Lumi 4B** para fazer *tool calling*: escolher e chamar as APIs certas. Hoje ele não faz nenhuma chamada válida. O melhor modelo vira a submissão do PDC ao SemEval-2027 Task 11 em janeiro.
- **Vence** quem tiver a maior taxa de acerto no **conjunto escondido (domínios fora do treino)**, nas **capacidades 1 e 2** da VAKRA, rodando no **agente oficial, sem alterar o código e com os mesmos parâmetros**.
- Livre: método de treino, dados de treino, uso de LLMs para gerar dados.

| Data | Marco | O que o harness entrega |
|---|---|---|
| Ter 10/11 | Largada | repositório, split, ambiente reproduzível |
| **Ter 17/11** | **Entrega 1:** baseline + análise de erros (pipeline de treino pronto na semana 1) | `make baseline` → `analyze` → `error_analysis.md` |
| 18–29/11 | Treino, avaliação no mesmo conjunto escondido | `run` + `eval` + `compare` a cada checkpoint |
| **Seg 30/11** | **Entrega 2:** modelo, Colab, post no blog | — (fora do harness) |

## 2. O que a VAKRA mede nas capacidades que valem

| | Capacidade 1 (`bi_apis`) | Capacidade 2 (`dashboard_apis`) |
|---|---|---|
| Tarefa | encadear APIs (aninhadas/composicionais) | escolher a ferramenta certa entre muitas |
| Treino (amostras / domínios) | 1.324 / 33 | 1.860 / 40 |
| Teste oficial (amostras / domínios) | 2.077 / 54 | 1.597 / 17 |
| Tool calls por amostra | ~4 | 1 |

- O agente oficial é um ReAct do LangGraph (`LangGraphReActAgent`) que fala com o ambiente por **MCP**, dentro de containers Docker. Defaults do runner: `--top-k-tools 128` (até 128 ferramentas no prompt, escolhidas por similaridade de embedding), temperatura 0, `AGENT_TIMEOUT_SECONDS=300`. **[lido]**
- O avaliador **reexecuta** as tool calls previstas e pontua só o **último turno**: (1) casamento exato das respostas esperadas das ferramentas → (2) juiz LLM de correção → (3) juiz LLM de *groundedness*. O juiz é `openai/gpt-oss-120b` via **Groq** (ou RITS, só IBM). **[lido]**

## 3. De onde o Lumi parte

- 4B, híbrido Mamba2 + atenção (`NemotronHForCausalLM`, 42 camadas), derivado do Nemotron 3 Nano 4B por CPT + destilação + SFT. PT-BR e EN. **[lido]**
- O cartão diz: *"Tool use is unvalidated"* e *"Thinking must remain disabled"*. Contexto validado: o CPT usou 8.192 tokens (o `config.json` declara 262.144, mas não é validado). **[lido]**
- O chat template já sabe tool calling, num formato XML estilo Qwen3-Coder:
  `<tool_call><function=nome><parameter=arg>valor</parameter></function></tool_call>`. No vLLM isso é `--tool-call-parser qwen3_coder`. **[lido; renderizei o template real em `make template`]**

## 4. Achados que mudam a execução

Em ordem de impacto. Cada um diz o que o harness faz a respeito.

1. **O dev precisa ser por domínio, e pela união das duas capacidades.** O escondido usa domínios fora do treino; um split por amostra superestimaria tudo. Os 33 domínios de treino da cap. 1 estão **todos** na cap. 2 (união = 40). Dev escolhido: `book_publishing_company, coinmarketcap, genes, image_and_language, law_episode, mental_health_survey, movie_platform, shakespeare` (cap. 1: 5 domínios; cap. 2: 8). → `configs/splits.json`, com teste de que nenhum domínio de dev vaza para o treino. **[dado lido]**

2. **O runner oficial só lê `data/test/`** e não cai para o treino. Aceita `CAPABILITY_1_DIR` / `CAPABILITY_2_DIR`. → `stage-dev` copia os domínios de dev para `work/dev/` e o `run` aponta o runner para lá, sem tocar em `data/test` nem no código. **[lido]**

3. **A capacidade 1 depende de um mapa de UUID → universo** (`environment/configs/mcp_tool_universe_id_mapping.yaml`, 2.085 UUIDs, gerado a partir da saída do *teste*). O UUID do exemplo `sample_data` da própria VAKRA não está nele, e por construção ele só cobre o teste, então esperamos o mesmo dos UUIDs de treino (a confirmar no `smoke`). UUID ausente faz o runner falhar com `Universe switch failed`. → `stage-dev` gera o mapa estendido com a mesma lógica do gerador oficial (testado contra o arquivo real: 2.085 + 1) e um `docker-compose.override.yml` que monta só essa pasta por cima do container da cap. 1. **[lido; funcionamento no container a confirmar no `make smoke`]**

4. **`enable_thinking` é uma armadilha.** O template assume `True` se o parâmetro não vier (abre `<think>`), o cartão exige desligado, e o agente oficial **não** passa o parâmetro. → o `serve` sobe o vLLM com `--default-chat-template-kwargs '{"enable_thinking": false}'`. **Risco que não controlamos:** se a organização servir o modelo sem essa flag, o modelo entregue vê `<think>`. Mitigação a decidir: entregar o modelo com o default do `chat_template.jinja` já em `False` (e/ou treinar para ser robusto aos dois prompts). **[lido; o aviso aparece em `make template`]**

5. **128 ferramentas no prompt × contexto validado de 8k.** Cada ferramenta ocupa várias linhas de XML no system prompt. Pode estourar o contexto ou degradar o modelo. → o proxy grava `prompt_tokens` e erros HTTP, e o relatório mostra p50/p90/máx e quantos passaram de 8.192. `max_model_len` está em 32.768 por precaução. **[a confirmar: ninguém mediu ainda]**

6. **O formato do dado de treino ≠ o formato do agente.** O GT da cap. 1 usa `initialize_active_data` e referências `$label.data_path$`; o agente em execução trabalha com *handles* (`data_label`) e um system prompt próprio. Converter o GT "à mão" pode treinar o modelo para um formato que o agente nunca apresenta. → o proxy grava, a cada chamada, exatamente `messages` + `tools` que o agente enviou e o que o modelo respondeu. Esse log é matéria-prima de SFT no formato exato do agente (rodar um modelo forte no agente oficial, filtrar pelo avaliador). **[lido; a conversão em si é o próximo passo]**

7. **O agente tem um *fallback* de texto, mas só na capacidade 2:** se o modelo terminar sem tool call nativa mas com um JSON `{"name":…,"arguments":…}`, o agente o executa (uma chamada só). Serve para a cap. 2 (1 chamada por amostra). Na cap. 1 ele é desligado (o código exige não haver "handle" inicial, e a cap. 1 sempre tem), então lá só vale tool call nativa. **[lido]**

8. **Sem chave Groq não há nota local.** O avaliador precisa de `API_KEY` (Groq) com `JUDGE_BACKEND=groq`. O custo/limite de taxa do `gpt-oss-120b` na Groq vai pesar nos ciclos de treino. → `eval` falha cedo com mensagem clara se faltar. **[lido]**

9. **Uma máquina, uma GPU (4090, 24 GB).** O agente oficial usa `docker exec` (MCP por stdio), então a mesma máquina precisa de Docker (~35 GB de disco; 8 GB+ de RAM no Docker porque o `compose up` sobe também a cap. 4, que usa ChromaDB) e da GPU. Consequências:
   - **vLLM não roda nativo no Windows.** Na 4090: Linux, WSL2 ou container (`make serve SERVE_ARGS=--docker`, imagem `vllm/vllm-openai` com `--gpus all`). **[a confirmar: o vLLM atual suporta o NemotronH do Lumi e o parser `qwen3_coder` nessa imagem]**
   - **Servir e treinar disputam a mesma GPU.** Pare o `serve` para treinar e suba de novo a cada checkpoint a avaliar. Fine-tuning completo de 4B provavelmente não cabe em 24 GB; LoRA/QLoRA cabe. Decisão para o pipeline de treino. **[a confirmar]**
   - Colab não roda Docker, então serve para a demo da Entrega 2, não para rodar o benchmark.

10. **O "conjunto escondido" não está definido no plano.** O mapa de universos (do teste oficial) contém **32 dos 33** domínios de treino da cap. 1. Ou seja, pelo menos na cap. 1 o teste oficial da VAKRA **não** é "fora do treino" em relação ao split público de treino (domínio repetido não quer dizer UUID repetido, mas o domínio já foi visto). Quem define o escondido do desafio? **[dado lido; pergunta abaixo]**

11. **Não consegui acessar `github.com/ceia-pdc-slms/desafio-lumi-agentico`** (privado, ou ainda não existe para a minha conta). Pode ter regras e parâmetros que o plano não traz.

12. **Licença do dataset:** CC-BY-NC-SA-4.0 (não comercial, compartilha igual). Relevante para o modelo publicado e para o post.

## 5. Perguntas para a organização (antes de 17/11)

1. Quais são exatamente os **parâmetros "iguais"**? Assumi os defaults do runner (`configs/official.json`): `--provider openai`, `--temperature 0.0`, `--top-k-tools 128`, sem `--max-iterations`, timeout 300 s.
2. **Como o modelo será servido** na avaliação (vLLM? flags? `enable_thinking`? `max_model_len`? parser de tool call?). Muda o que precisamos treinar.
3. **Qual é o conjunto escondido** (quais domínios) e **o que podemos usar do `train`** (todo ou um subconjunto)?
4. Quem fornece a **chave do juiz** (Groq) e com que limite?
5. Podemos acessar o repositório do desafio?

## 6. Plano de execução

**Semana 1 (até 17/11)**
1. `make setup` · `make vakra-clone` · `make vakra-data` · `make vakra-up` (na máquina da 4090, com Docker).
2. `make stage` (dev + mapa de universos) e reiniciar a cap. 1.
3. `make serve` (ou `make serve SERVE_ARGS=--docker` no Windows) num terminal e `make proxy RUN_ID=baseline-1` em outro.
4. `make smoke RUN_ID=baseline-1` → corrige o encanamento (UUID, checksum, template, parser) com 1 pergunta por domínio.
5. `make baseline` → `make eval` → `make analyze`. O `error_analysis.md` é o corpo da Entrega 1.
6. Em paralelo: pipeline de treino (SFT a partir dos traces + GT validado).

**18–29/11:** cada checkpoint entra em `run` + `eval` + `compare baseline <run>`. Mesmo dev e mesmos parâmetros em todas as execuções, para os números serem comparáveis.

**30/11:** modelo, Colab e post. O post pode usar direto as tabelas do `analyze`.

## 7. O que o harness cobre e o que não

- **Verificado aqui (30 testes, sem GPU/Docker):** split por domínio sem vazamento; mapa de universos (contra o arquivo real de 37 mil linhas e contra os dois formatos de `sequence`); classificação de falhas; proxy (repasse idêntico, gravação, tools deduplicadas, 502 se o upstream cair); template real do Lumi; comando do vLLM; que nenhuma `OPENAI_API_KEY` real é enviada ao endpoint do Lumi.
- **Não executado de ponta a ponta** (falta GPU, ~35 GB e Docker): subir vLLM com o parser `qwen3_coder` no NemotronH, rodar o `benchmark_runner.py` e o avaliador de verdade, o override do compose, e o checksum de ferramentas em domínio de treino (o código só avisa, não falha). O `make smoke` existe para expor isso no primeiro dia.
- **Fora do escopo desta entrega:** o pipeline de treino e as capacidades 3 e 4 (não contam para o vencedor).
