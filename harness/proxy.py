"""Proxy de gravação OpenAI-compatível entre o agente oficial e o vLLM.

O agente LangGraph esconde o que o modelo realmente devolveu: se o Lumi escreve um
<tool_call> que o parser do vLLM não reconhece, o agente só vê "nenhuma tool call".
Gravando cada requisição/resposta aqui, a análise de erros enxerga o texto bruto, o
tamanho do prompt (estouro de contexto com 128 ferramentas?) e os erros HTTP. O mesmo
log vira matéria-prima para dados de treino no formato exato do agente.

Uso: o agente aponta para o proxy (LUMI_BASE_URL=http://localhost:8001/v1) e o proxy
repassa para o vLLM. Não altera nada do que passa. Respostas em streaming são
bufferizadas (o benchmark não depende de streaming incremental).
"""
from __future__ import annotations

import hashlib
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade", "content-length", "host"}


class TraceWriter:
    """JSONL thread-safe; tools (grandes e repetidas) vão para arquivos deduplicados por hash."""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.tools_dir = self.log_path.with_suffix(".tools")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def tools_ref(self, tools: list) -> str:
        blob = json.dumps(tools, sort_keys=True, ensure_ascii=False)
        ref = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
        path = self.tools_dir / f"{ref}.json"
        with self._lock:
            if not path.exists():
                path.write_text(blob, encoding="utf-8")
        return ref

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def _make_handler(upstream: str, writer: TraceWriter, timeout: float):
    up = urlparse(upstream)
    prefix = up.path.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silencioso
            pass

        def _forward(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            req_json = None
            if body and self.path.endswith("/chat/completions"):
                try:
                    req_json = json.loads(body)
                except json.JSONDecodeError:
                    req_json = None

            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            conn_cls = http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
            started = time.perf_counter()
            status, resp_body, resp_headers, error = 502, b"", [], None
            try:
                conn = conn_cls(up.hostname, up.port, timeout=timeout)
                conn.request(self.command, prefix + self.path, body=body or None, headers=headers)
                resp = conn.getresponse()
                status, resp_body = resp.status, resp.read()
                resp_headers = [(k, v) for k, v in resp.getheaders() if k.lower() not in HOP_BY_HOP]
                conn.close()
            except Exception as exc:  # noqa: BLE001 - qualquer falha de rede vira 502 + registro
                error = f"{type(exc).__name__}: {exc}"
                resp_body = json.dumps({"error": {"message": error, "type": "proxy_upstream_error"}}).encode()
                resp_headers = [("Content-Type", "application/json")]
            latency = time.perf_counter() - started

            if req_json is not None:
                writer.write(self._record(req_json, status, resp_body, latency, error))

            self.send_response(status)
            for k, v in resp_headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

        def _record(self, req: dict, status: int, resp_body: bytes, latency: float, error: str | None) -> dict:
            tools = req.get("tools") or []
            request = {k: v for k, v in req.items() if k not in ("messages", "tools")}
            request["messages"] = req.get("messages", [])
            request["n_tools"] = len(tools)
            if tools:
                request["tools_ref"] = writer.tools_ref(tools)
            try:
                response = json.loads(resp_body) if resp_body else None
            except json.JSONDecodeError:
                response = {"raw": resp_body.decode("utf-8", "replace")[:4000]}
            return {
                "ts": time.time(),
                "path": self.path,
                "status": status,
                "latency_s": round(latency, 4),
                "error": error,
                "request": request,
                "response": response,
            }

        do_POST = _forward
        do_GET = _forward

    return Handler


def make_server(upstream: str, log_path: Path, host: str = "127.0.0.1", port: int = 8001,
                timeout: float = 600.0) -> ThreadingHTTPServer:
    writer = TraceWriter(log_path)
    server = ThreadingHTTPServer((host, port), _make_handler(upstream, writer, timeout))
    server.daemon_threads = True
    return server


def serve(upstream: str, log_path: Path, host: str, port: int) -> None:
    server = make_server(upstream, log_path, host, port)
    print(f"proxy {host}:{server.server_address[1]} -> {upstream}\n  log: {log_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
