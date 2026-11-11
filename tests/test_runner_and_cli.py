import json
import os

import pytest

from harness import __main__ as cli
from harness import runner, template_check
from harness.paths import DEV_DIR, ROOT, load_env_file, official_config


def test_env_never_forwards_a_real_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-REAL-SECRET")
    monkeypatch.delenv("LUMI_API_KEY", raising=False)
    env = runner.build_env(official_config(), "http://gpu:8001/v1", "dev", [1, 2])
    assert env["OPENAI_API_KEY"] == "EMPTY"
    assert "sk-REAL-SECRET" not in env.values()


def test_env_points_runner_at_staged_dev_dirs():
    env = runner.build_env(official_config(), "http://gpu:8001/v1", "dev", [1, 2])
    assert env["OPENAI_BASE_URL"] == env["OPENAI_API_BASE"] == "http://gpu:8001/v1"
    assert env["CAPABILITY_1_DIR"] == str((DEV_DIR / "capability_1_bi_apis").resolve())
    assert env["CAPABILITY_2_DIR"] == str((DEV_DIR / "capability_2_dashboard_apis").resolve())
    assert env["AGENT_TIMEOUT_SECONDS"] == "300"


def test_run_command_uses_only_official_parameters(tmp_path):
    cfg = official_config()
    cmd = runner.build_run_command(2, tmp_path / "out", cfg, "python", max_samples=2, domains=["genes"])
    assert cmd[:2] == ["python", "benchmark_runner.py"]
    flat = " ".join(cmd)
    for expected in ("--provider openai", "--model lumi-4b", "--temperature 0.0", "--top-k-tools 128",
                     "--max-samples-per-domain 2", "--domain genes"):
        assert expected in flat
    assert "--max-iterations" not in cmd  # default oficial: não passar
    assert cmd[cmd.index("--output") + 1] == str((tmp_path / "out").resolve())


def test_serve_command_has_tool_calling_and_thinking_off():
    flat = " ".join(cli.vllm_command())
    assert "--enable-auto-tool-choice" in flat and "--tool-call-parser qwen3_coder" in flat
    assert '{"enable_thinking": false}' in flat
    assert "--revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d" in flat


def test_serve_docker_command_for_windows_4090():
    cmd = cli.vllm_command(docker=True)
    assert cmd[:2] == ["docker", "run"] and "--gpus" in cmd and "--ipc=host" in cmd
    image = cmd.index("vllm/vllm-openai:latest")
    # tudo que é do vLLM vem DEPOIS da imagem (o entrypoint da imagem já é `vllm serve`)
    assert cmd[image + 1] == "ceia-br/lumi-4b" and "vllm" not in cmd[image + 1:image + 2]
    assert "--tool-call-parser" in cmd[image:] and "--enable-auto-tool-choice" in cmd[image:]
    # mesmos argumentos do modo nativo: o comportamento do modelo não pode depender de como sobe
    assert cmd[image + 1:] == cli.vllm_command()[2:]


def test_cli_parser_accepts_every_command():
    p = cli.build_parser()
    for argv in (["splits", "--source", "x"], ["stage-dev"], ["run", "--tag", "t"], ["eval", "r"], ["analyze", "r"],
                 ["compare", "a", "b"], ["proxy"], ["check-template"], ["serve", "--print"],
                 ["serve", "--docker", "--print"]):
        assert p.parse_args(argv).fn is not None


def test_env_file_parsing_and_no_override(tmp_path, monkeypatch):
    for k in ("API_KEY", "JUDGE_BACKEND", "QUOTED", "KEEP", "LUMI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KEEP", "do-ambiente")
    f = tmp_path / ".env"
    f.write_text(
        "# comentario\n\nAPI_KEY=            # vazio com comentario\nJUDGE_BACKEND=groq   # inline\n"
        "QUOTED='a # b'\nKEEP=do-arquivo\nLUMI_BASE_URL=http://gpu:8001/v1\n", encoding="utf-8")

    loaded = load_env_file(f)

    assert loaded == {"JUDGE_BACKEND": "groq", "QUOTED": "a # b", "LUMI_BASE_URL": "http://gpu:8001/v1"}
    assert "API_KEY" not in os.environ          # vazio não vira "# vazio com comentario"
    assert os.environ["KEEP"] == "do-ambiente"  # o ambiente vence o arquivo
    monkeypatch.delenv("JUDGE_BACKEND"); monkeypatch.delenv("QUOTED"); monkeypatch.delenv("LUMI_BASE_URL")


def test_env_example_is_loadable_and_does_not_set_secrets(monkeypatch):
    for k in ("HF_TOKEN", "API_KEY", "JUDGE_BACKEND", "LUMI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    loaded = load_env_file(ROOT / ".env.example")
    assert "API_KEY" not in loaded and "HF_TOKEN" not in loaded
    assert loaded["JUDGE_BACKEND"] == "groq"
    for k in loaded:
        monkeypatch.delenv(k, raising=False)


# --- template -----------------------------------------------------------------------------
MINI = "{% if enable_thinking is defined and not enable_thinking %}<think></think>{% endif %}{{ messages[0]['content'] }}"


def test_template_render_passes_kwargs():
    assert template_check.render(MINI, [{"role": "user", "content": "oi"}], enable_thinking=False) == "<think></think>oi"


def test_template_check_reports_error_for_unusable_template():
    findings, _ = template_check.check("{{ raise_exception('sem suporte a tools') }}")
    assert findings[0].level == "erro"


@pytest.mark.network
def test_real_lumi_template_supports_tool_calling():
    cfg = official_config()["lumi"]
    path = template_check.fetch_template(cfg["hf_repo"], cfg["revision"])
    findings, _ = template_check.check(path.read_text(encoding="utf-8"))
    assert not [f for f in findings if f.level == "erro"], findings
