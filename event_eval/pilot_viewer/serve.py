"""Serve the dynamic annotation viewer. Stdlib only.

    python serve.py --port 8000

Endpoints:
    /                     the viewer app (index.html from this directory)
    /read?path=<p>        return a text file (annotation JSONL); relative
                          paths resolve against the repo root (two levels up)
    /video?root=<r>&rel=<v>  byte-range video streaming from <root>/<rel>
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

_VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".m4v"}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/read":
            return self._read_file(query.get("path", ""))
        if url.path == "/video":
            return self._stream_video(query.get("root", ""), query.get("rel", ""))
        rel = url.path.lstrip("/") or "index.html"
        target = (HERE / rel).resolve()
        if not str(target).startswith(str(HERE)) or not target.is_file():
            return self.send_error(404)
        return self._send_file(target, ranged=False)

    def _read_file(self, raw: str):
        if not raw:
            return self._json_error(400, "missing path")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        path = path.resolve()
        if path.suffix.lower() not in {".json", ".jsonl"}:
            return self._json_error(403, "only .json/.jsonl files can be read")
        if not path.is_file():
            return self._json_error(404, f"not found: {path}")
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream_video(self, root: str, rel: str):
        if not root or not rel:
            return self._json_error(400, "missing root or rel")
        base = Path(root).expanduser().resolve()
        target = (base / rel).resolve()
        if not str(target).startswith(str(base)):
            return self._json_error(403, "path escapes root")
        if target.suffix.lower() not in _VIDEO_EXTS:
            return self._json_error(403, "not a video file")
        if not target.is_file():
            return self._json_error(404, f"not found: {target}")
        return self._send_file(target, ranged=True)

    def _send_file(self, target: Path, *, ranged: bool):
        size = target.stat().st_size
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        start, end = 0, size - 1
        status = 200
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "")) if ranged else None
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
            else:
                start = size - int(m.group(2))
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with target.open("rb") as fh:
            fh.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def _json_error(self, code: int, message: str):
        body = json.dumps({"error": message}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"Viewer on http://localhost:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
