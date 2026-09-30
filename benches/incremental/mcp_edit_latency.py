#!/usr/bin/env python3
"""Edit-then-search latency through one `rfx mcp` stdio session (auto_update.sh).

    mcp_edit_latency.py <rfx> <file relative to cwd> <runs>

One warm search, then per run: append a token line to <file>, wait past the 1 s
freshness memo, and time a `search_code` for the token. Prints each request's
latency, whether the answer was fresh and found the token, and the server's peak
RSS (VmHWM). The file is left edited; the caller reverts it.
"""
import json
import subprocess
import sys
import time

rfx, path, runs = sys.argv[1], sys.argv[2], int(sys.argv[3])
proc = subprocess.Popen(
    [rfx, "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
)
next_id = 0


def call(tool, args):
    global next_id
    next_id += 1
    req = {"jsonrpc": "2.0", "id": next_id, "method": "tools/call",
           "params": {"name": tool, "arguments": args}}
    start = time.perf_counter()
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    ms = (time.perf_counter() - start) * 1000
    body = json.loads(json.loads(line)["result"]["content"][0]["text"])
    return ms, body


def peak_kb():
    with open(f"/proc/{proc.pid}/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


ms, _ = call("search_code", {"pattern": "NewController", "limit": 20})
print(f"mcp_warm    {ms:8.1f} ms")
original = open(path, "rb").read()
for i in range(runs):
    token = f"zz_mcp_probe_{i}_{int(time.time())}"
    with open(path, "ab") as f:
        f.write(f"// {token}\n".encode())
    time.sleep(1.2)  # past the freshness memo, as an agent's next turn would be
    ms, body = call("search_code", {"pattern": token})
    found = len(body.get("rows", body.get("results", [])))
    print(f"mcp_edit    {ms:8.1f} ms  status={body.get('status')}  found={found}  "
          f"load={open('/proc/loadavg').read().split()[0]}")
    with open(path, "wb") as f:
        f.write(original)
    time.sleep(1.2)
    call("search_code", {"pattern": "NewController", "limit": 1})  # the revert
print(f"mcp_peak    rss={peak_kb()} KB")
proc.stdin.close()
proc.wait(timeout=30)
