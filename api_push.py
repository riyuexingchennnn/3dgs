#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""当 github.com 不可达时，用 Git Data API 把 docs 产物与 sphinx 源推送到仓库。"""
import base64
import json
import os
import subprocess
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

REPO = "riyuexingchennnn/3dgs"
BRANCH = "master"
MSG = "docs: 正文宽度对齐 ROS 2 文档(64rem)，正文链接统一为蓝色"
ENV = dict(os.environ, GODEBUG="http2client=0")


def log(*a):
    print(*a, flush=True)


def gh(*args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, env=ENV)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def get(path):
    rc, out, err = gh("api", path)
    return json.loads(out) if rc == 0 else None


def post(path, payload, tag=""):
    tf = f"/tmp/gh_{tag or 'payload'}.json"
    with open(tf, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    last = ""
    for attempt in range(5):
        rc, out, err = gh("api", "--method", "POST", path, "--input", tf)
        if rc == 0:
            return json.loads(out)
        last = err
        time.sleep(2)
    raise RuntimeError(f"{path} 失败: {last[:200]}")


def main():
    parent = get(f"repos/{REPO}/git/ref/heads/{BRANCH}")["object"]["sha"]
    base_tree = get(f"repos/{REPO}/git/commits/{parent}")["tree"]["sha"]
    log(f"远程 parent: {parent[:8]}")

    targets = [f"docs/{f}" for f in sorted(os.listdir("docs")) if f.endswith(".html")]
    targets += [
        "docs/_static/custom.css",
        "docs/_static/logo.jpg",
        "docs/_static/favicon.png",
        "sphinx/conf.py",
        "sphinx/_static/custom.css",
        "sphinx/_static/logo.jpg",
        "sphinx/_static/favicon.png",
    ]

    entries = []
    for i, p in enumerate(targets):
        content = base64.b64encode(open(p, "rb").read()).decode()
        blob = post(f"repos/{REPO}/git/blobs", {"content": content, "encoding": "base64"}, tag=f"b{i}")
        entries.append({"path": p, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        log(f"  [{i+1}/{len(targets)}] {p}")

    for p in ["docs/_static/custom.js", "sphinx/_static/custom.js"]:
        if get(f"repos/{REPO}/contents/{p}?ref={BRANCH}"):
            entries.append({"path": p, "mode": "100644", "type": "blob", "sha": None})
            log("  标记删除:", p)

    tree = post(f"repos/{REPO}/git/trees", {"base_tree": base_tree, "tree": entries}, tag="tree")
    log(f"新 tree: {tree['sha'][:8]}")
    commit = post(f"repos/{REPO}/git/commits",
                  {"message": MSG, "tree": tree["sha"], "parents": [parent]}, tag="commit")
    log(f"新 commit: {commit['sha'][:8]}")
    rc, out, err = gh("api", "--method", "PATCH", f"repos/{REPO}/git/refs/heads/{BRANCH}",
                      "-f", f"sha={commit['sha']}")
    log("master ->", json.loads(out)["object"]["sha"][:8] if rc == 0 else err[:200])
    log("完成")


if __name__ == "__main__":
    main()
