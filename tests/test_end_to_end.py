import subprocess
import json
import sys
import time
from pathlib import Path
import os
import pytest

@pytest.fixture(scope="module")
def build_index(tmp_path_factory):
    # use existing notes in repo
    repo_root = Path.cwd()
    notes_dir = repo_root / "notes"
    if not notes_dir.exists():
        pytest.skip("notes folder not present")
    index = repo_root / "notes_index.npz"
    meta = repo_root / "notes_meta.json"
    # run build_index.py
    cmd = [sys.executable, "build_index.py", str(notes_dir), "--index", str(index), "--meta", str(meta)]
    subprocess.run(cmd, check=True)
    assert index.exists() and meta.exists()
    return str(index), str(meta)

def test_search_and_list(build_index):
    index, meta = build_index
    cmd = [sys.executable, "notes_mcp_server.py", "--index", index, "--meta", meta, "--notes_root", "./notes"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    time.sleep(1.0)
    try:
        # list_notes
        req = {"jsonrpc":"2.0","id":"1","method":"list_notes","params":{}}
        payload = json.dumps(req).encode("utf-8")
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
        proc.stdin.write(header + payload)
        proc.stdin.flush()
        # read response
        header_bytes = b""
        while True:
            line = proc.stdout.readline()
            header_bytes += line
            if header_bytes.endswith(b"\r\n\r\n") or header_bytes.endswith(b"\n\n"):
                break
        import re
        m = re.search(rb"Content-Length:\s*(\d+)", header_bytes, re.I)
        assert m, "No Content-Length in response header"
        length = int(m.group(1))
        body = proc.stdout.read(length)
        resp = json.loads(body.decode("utf-8"))
        assert "result" in resp
        # search
        req2 = {"jsonrpc":"2.0","id":"2","method":"search_notes","params":{"query":"PCA","top_k":2}}
        payload2 = json.dumps(req2).encode("utf-8")
        header2 = f"Content-Length: {len(payload2)}\r\n\r\n".encode("ascii")
        proc.stdin.write(header2 + payload2)
        proc.stdin.flush()
        header_bytes = b""
        while True:
            line = proc.stdout.readline()
            header_bytes += line
            if header_bytes.endswith(b"\r\n\r\n") or header_bytes.endswith(b"\n\n"):
                break
        m2 = re.search(rb"Content-Length:\s*(\d+)", header_bytes, re.I)
        assert m2
        length2 = int(m2.group(1))
        body2 = proc.stdout.read(length2)
        resp2 = json.loads(body2.decode("utf-8"))
        assert "result" in resp2
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
