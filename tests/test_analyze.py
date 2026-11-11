import json

from harness import analyze as an

TOOLS = ["select_data_equal_to", "compute_data_count"]


def rec(uuid, *, status="success", error="", calls=(), answer="resposta"):
    return {"uuid": uuid, "domain": "genes", "status": status, "error": error, "duration_s": 1.5,
            "output": [{"turn_id": 0, "query": f"q-{uuid}", "answer": answer,
                        "sequence": {"tool_call": [{"name": n, "arguments": {}} for n in calls]}}]}


def test_classification_rules_in_priority_order():
    c = an.classify
    assert c(rec("a", status="error", error="Agent timed out after 300.0 seconds"), TOOLS, None)[0] == "error_timeout"
    assert c(rec("a", status="error", error="BadRequestError maximum context length is 8192"), TOOLS, None)[0] == "error_context_overflow"
    assert c(rec("a", status="error", error="RuntimeError Universe switch failed: x"), TOOLS, None)[0] == "error_universe_switch"
    assert c(rec("a", status="error", error="APIConnectionError boom"), TOOLS, None)[0] == "error_connection"
    assert c(rec("a", status="error", error="ValueError x"), TOOLS, None)[0] == "error_other"
    assert c(rec("a", answer="vou chamar"), TOOLS, None)[0] == "no_tool_call"
    assert c(rec("a", answer="<tool_call>\n<function=x>"), TOOLS, None)[0] == "no_tool_call_markup"
    assert c(rec("a", calls=["inventada"]), TOOLS, None)[0] == "unknown_tool"
    assert c(rec("a", calls=["compute_data_count"], answer=" "), TOOLS, None)[0] == "empty_answer"
    assert c(rec("a", calls=["compute_data_count"]), TOOLS, None)[0] == "unscored"
    assert c(rec("a", calls=["compute_data_count"]), TOOLS, 1.0)[0] == "correct"
    assert c(rec("a", calls=["compute_data_count"]), TOOLS, 0.0)[0] == "wrong_answer"


def test_valid_call_means_known_tool_name():
    _, extras = an.classify(rec("a", calls=["inventada", "compute_data_count"]), TOOLS, None)
    assert extras["valid_call"] and extras["n_calls"] == 2
    _, extras = an.classify(rec("a", calls=["inventada"]), TOOLS, None)
    assert not extras["valid_call"]


def test_sequence_as_list_is_supported():
    r = rec("a")
    r["output"][0]["sequence"] = [{"tool_call": [{"name": "compute_data_count", "arguments": {}}]}]
    assert an.classify(r, TOOLS, 1.0)[0] == "correct"


def _make_run(tmp_path):
    run = tmp_path / "20261110-1200-baseline"
    cap = run / "capability_2_dashboard_apis"
    cap.mkdir(parents=True)
    records = [rec("u1", calls=["compute_data_count"]), rec("u2"), rec("u3", status="error", error="timed out"),
               rec("u4", calls=["inventada"])]
    (cap / "genes.json").write_text(json.dumps(records), encoding="utf-8")
    (cap / "genes_tools.json").write_text(json.dumps(
        [{"uuid": r["uuid"], "domain": "genes", "query": "q", "all_tools": TOOLS, "shortlisted_tools": TOOLS} for r in records]), encoding="utf-8")
    (run / "eval").mkdir()
    (run / "eval" / "capability_2.results.json").write_text(json.dumps({"domains": {"genes": {"dialogues": [
        {"uuid": "u1", "score": 1.0}, {"uuid": "u2", "score": 0.0}]}}}), encoding="utf-8")
    return run


def test_end_to_end_report(tmp_path):
    run = _make_run(tmp_path)
    summary = an.analyze(run)

    o = summary["overall"]
    assert o["n"] == 4
    assert o["valid_call_rate"] == 0.25          # só u1 tem chamada válida
    assert o["accuracy"] == 0.25                 # 1 correta em 4; as 2 sem nota contam como erro
    assert o["scored"] == 2
    assert o["categories"] == {"correct": 1, "no_tool_call": 1, "error_timeout": 1, "unknown_tool": 1}
    md = (run / "error_analysis.md").read_text(encoding="utf-8")
    assert "chamada válida" in md and "`error_timeout`" in md and "q-u3" in md
    assert (run / "analysis.json").exists()


def test_proxy_stats(tmp_path):
    log = tmp_path / "proxy.jsonl"
    ok = lambda content, tool_calls, prompt: {  # noqa: E731
        "status": 200, "request": {"n_tools": 128},
        "response": {"usage": {"prompt_tokens": prompt, "completion_tokens": 20},
                     "choices": [{"finish_reason": "stop", "message": {"content": content, "tool_calls": tool_calls}}]}}
    lines = [ok("", [{"id": "1"}], 9000), ok("<tool_call>\n<function=x>", None, 7000), ok("oi", None, 3000),
             {"status": 400, "request": {"n_tools": 128}, "response": {"error": {"message": "maximum context length is 8192"}}}]
    log.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")

    s = an.proxy_stats(log)

    assert s["requests"] == 4 and s["http_status"] == {"200": 3, "400": 1}
    assert s["responses_with_tool_calls"] == 1
    assert s["responses_with_markup_but_no_tool_calls"] == 1
    assert s["context_overflow_errors"] == 1
    assert s["prompt_over_8192"] == 1 and s["prompt_tokens"]["max"] == 9000
    assert s["tools_in_prompt"]["max"] == 128
    assert an.proxy_stats(tmp_path / "nao_existe.jsonl") == {}
