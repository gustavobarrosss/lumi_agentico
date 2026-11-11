import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from harness import proxy

TOOLS = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}}]


class FakeVLLM(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVLLM.seen.append((self.path, body, self.headers.get("Authorization")))
        if body.get("model") == "falha":
            payload, code = {"error": {"message": "maximum context length is 8192"}}, 400
        else:
            payload, code = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                             "usage": {"prompt_tokens": 11, "completion_tokens": 2}}, 200
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _start(server):
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t


def _post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_proxy_forwards_unchanged_and_records(tmp_path):
    FakeVLLM.seen.clear()
    upstream = HTTPServer(("127.0.0.1", 0), FakeVLLM)
    _start(upstream)
    log = tmp_path / "proxy.jsonl"
    prox = proxy.make_server(f"http://127.0.0.1:{upstream.server_address[1]}", log, port=0)
    _start(prox)
    port = prox.server_address[1]
    try:
        body = {"model": "lumi-4b", "messages": [{"role": "user", "content": "oi"}], "tools": TOOLS, "temperature": 0.0}
        status, resp = _post(port, body)
        assert status == 200 and resp["choices"][0]["message"]["content"] == "ok"
        # o upstream recebeu exatamente o que o cliente mandou, inclusive o cabeçalho de auth
        path, seen, auth = FakeVLLM.seen[0]
        assert path == "/v1/chat/completions" and seen == body and auth == "Bearer EMPTY"

        status, resp = _post(port, {**body, "model": "falha"})  # erro do upstream é repassado e gravado
        assert status == 400 and "maximum context" in resp["error"]["message"]
        _post(port, body)  # mesmas tools de novo: arquivo deduplicado
    finally:
        prox.shutdown()
        upstream.shutdown()

    recs = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in recs] == [200, 400, 200]
    assert recs[0]["request"]["messages"] == body["messages"]
    assert recs[0]["request"]["n_tools"] == 1 and "tools" not in recs[0]["request"]
    assert recs[0]["response"]["usage"]["prompt_tokens"] == 11
    assert recs[0]["latency_s"] >= 0
    # tools gravadas uma vez só, recuperáveis pelo hash
    files = list(log.with_suffix(".tools").glob("*.json"))
    assert len(files) == 1 and recs[0]["request"]["tools_ref"] == files[0].stem
    assert json.loads(files[0].read_text(encoding="utf-8")) == TOOLS


def test_proxy_upstream_down_returns_502_and_records(tmp_path):
    log = tmp_path / "p.jsonl"
    prox = proxy.make_server("http://127.0.0.1:1", log, port=0, timeout=2)  # porta 1: ninguém escutando
    _start(prox)
    try:
        status, resp = _post(prox.server_address[1], {"model": "m", "messages": []})
    finally:
        prox.shutdown()
    assert status == 502 and resp["error"]["type"] == "proxy_upstream_error"
    rec = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert rec["status"] == 502 and rec["error"]
