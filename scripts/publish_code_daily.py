#!/usr/bin/env python3
"""Review-gated, code-only snapshots into a separate Git branch (stdlib only)."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

DENIED_PARTS = {"workspace", "weights", "checkpoints", "datasets", "reports", ".git", ".ssh", "__pycache__"}
DENIED_SUFFIXES = {".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".gguf", ".npz", ".npy", ".xlsx", ".zip", ".tar", ".gz", ".log", ".jsonl", ".pem", ".key", ".pyc"}
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".toml", ".txt", ".ir"}
SPECIAL_NAMES = {".gitignore", ".gitattributes"}
MAX_BYTES = 2 * 1024 * 1024
SECRET_PATTERNS = (
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
    r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}",
    r"\bsk-(?:proj-)?[A-Za-z0-9_-]{35,}",
    r"https?://[^/@\s]+:[^/@\s]+@",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def validate_path(name, role):
    p = PurePosixPath(name)
    if not name or "\\" in name or p.is_absolute() or ".." in p.parts or str(p) != name:
        raise ValueError("Unsafe relative path: " + name)
    if any(x.lower() in DENIED_PARTS or x.lower().startswith(".env") for x in p.parts):
        raise ValueError("Artifact/secret directory forbidden: " + name)
    if p.suffix.lower() in DENIED_SUFFIXES or (p.suffix.lower() not in TEXT_SUFFIXES and name not in SPECIAL_NAMES):
        raise ValueError("Non-code file type: " + name)
    if role not in {"runtime_code", "runtime_config", "verification_code", "publication_policy", "documentation"}:
        raise ValueError("Unreviewed role: " + name)
    if p.suffix.lower() == ".ir" and role != "runtime_code":
        raise ValueError("Operator graph must be reviewed runtime source: " + name)
    if p.suffix.lower() == ".json" and role not in {"runtime_config", "publication_policy"}:
        raise ValueError("JSON requires a real configuration consumer: " + name)
    return p


def validate_content(name, data, entry=None):
    if len(data) > MAX_BYTES or b"\x00" in data:
        raise ValueError("Binary/oversized content: " + name)
    text = data.decode("utf-8-sig")
    if any(re.search(p, text) for p in SECRET_PATTERNS):
        raise ValueError("Credential marker; value suppressed: " + name)
    if name.endswith(".py"):
        tree = ast.parse(text, filename=name)
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)) and len(n.value) > 32768:
                raise ValueError("Embedded payload requires separate review: " + name)
            pairs = []
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant):
                pairs = [(t.id, n.value.value) for t in n.targets if isinstance(t, ast.Name)]
            if isinstance(n, ast.Dict):
                pairs += [(k.value, v.value) for k, v in zip(n.keys, n.values)
                          if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)]
            for key, value in pairs:
                if isinstance(key, str) and key.lower() in {"password", "api_key", "access_token", "auth_token", "secret_key", "openai_api_key"}:
                    if isinstance(value, str) and len(value) >= 4 and value.upper() not in {"EMPTY", "NONE", "DUMMY", "PLACEHOLDER", "YOUR_API_KEY"} and not value.startswith(("${", "<", "{")):
                        raise ValueError("Literal credential assignment; value suppressed: " + name)
    elif name.endswith(".ir"):
        approved = (entry or {}).get("weight_free_operator_graph_SHA256", [])
        if digest(data) not in approved or not text.lstrip().startswith("graph("):
            raise ValueError("Operator graph needs explicit weight-free source review: " + name)
        for line in text.splitlines():
            if "prim::Constant" in line and re.search(r"Tensor|BFloat|Half\(|value=\{", line):
                raise ValueError("Tensor constants forbidden in operator source: " + name)
    elif name.endswith(".json"):
        json.loads(text)


def load_policy(path):
    raw = path.read_bytes()
    policy = json.loads(raw)
    if policy["repository"] != "https://github.com/lyinguo/910b.git":
        raise ValueError("Unexpected publication destination")
    if policy["branch"] != "codex/910b-source-layout-20261114":
        raise ValueError("Unexpected publication branch")
    entries = policy["files"]
    if not isinstance(entries, dict) or not entries:
        raise ValueError("An explicit reviewed path list is required")
    for name, entry in entries.items():
        validate_path(name, entry["role"])
        if not entry.get("reason"):
            raise ValueError("Missing inclusion reason: " + name)
    return policy, digest(raw)


def git(repo, *args, check=True):
    p = subprocess.run(["git", "-c", "credential.interactive=never", *args], cwd=repo,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode:
        raise RuntimeError("Git operation failed: " + (p.stderr + p.stdout).decode("utf-8", errors="replace")[-2000:])
    return p


def checked_file(root, name):
    path = root / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Missing/linked/outside source: " + name)
    return path.read_bytes()


def plan_snapshot(source, repo, policy_path):
    policy, policy_sha = load_policy(policy_path)
    branch = git(repo, "symbolic-ref", "--short", "HEAD").stdout.decode().strip()
    if branch != policy["branch"] or git(repo, "remote", "get-url", "origin").stdout.decode().strip() != policy["repository"]:
        raise ValueError("Branch or remote differs from reviewed policy")
    status = git(repo, "status", "--porcelain=v1", "-z").stdout
    if status:
        raise ValueError("Publication checkout must be clean before planning")
    head_p = git(repo, "rev-parse", "--verify", "HEAD", check=False)
    head = head_p.stdout.decode().strip() if head_p.returncode == 0 else None
    records = []
    for name, entry in sorted(policy["files"].items()):
        data = checked_file(source, name)
        validate_content(name, data, entry)
        previous = (repo / name).read_bytes() if (repo / name).is_file() else None
        records.append(dict(path=name, SHA256=digest(data), bytes=len(data), role=entry["role"],
                            action="unchanged" if previous == data else ("update" if previous is not None else "add")))
    tracked = set(filter(None, git(repo, "ls-files", "-z").stdout.decode().split("\0")))
    if tracked - policy["files"].keys():
        raise ValueError("Checkout contains files outside the explicit code policy")
    plan = dict(branch=branch, repository=policy["repository"], HEAD=head, policy_SHA256=policy_sha,
                files=records, changes=sum(r["action"] != "unchanged" for r in records))
    plan["plan_SHA256"] = digest(canonical(plan))
    return plan


def audit_outgoing(repo, policy, base):
    revision = (base + "..HEAD") if base else "HEAD"
    commits = git(repo, "rev-list", revision).stdout.decode().splitlines()
    for commit in commits:
        paths = git(repo, "ls-tree", "-r", "--name-only", "-z", commit).stdout.decode().split("\0")
        for name in filter(None, paths):
            if name not in policy["files"]:
                raise ValueError("Outgoing commit contains an unapproved file: " + name)
            validate_path(name, policy["files"][name]["role"])
            data = git(repo, "show", commit + ":" + name).stdout
            validate_content(name, data, policy["files"][name])
    return commits


def publish(source, repo, policy_path, expected_plan_sha, message):
    lock = repo / ".git" / "code-publication.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode()); os.close(fd); fd = None
        policy, _ = load_policy(policy_path)
        plan = plan_snapshot(source, repo, policy_path)
        if plan["plan_SHA256"] != expected_plan_sha:
            raise ValueError("Candidate changed after review; generate a new plan")
        today = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        if not plan["changes"]:
            return dict(status="no_changes", commit=plan["HEAD"], day=today)
        if plan["HEAD"]:
            messages = git(repo, "log", "-200", "--format=%B%x00").stdout.decode()
            if "Code-Day: " + today in messages:
                raise ValueError("A code commit already exists for this Beijing day")
        remote = git(repo, "ls-remote", "origin", "refs/heads/" + policy["branch"]).stdout.decode().strip()
        base = remote.split()[0] if remote else None
        if base:
            git(repo, "merge-base", "--is-ancestor", base, "HEAD")
        elif plan["HEAD"]:
            # The complete outgoing ancestry is audited below; no imported Gitee history.
            pass
        changed = [r for r in plan["files"] if r["action"] != "unchanged"]
        for rec in changed:
            data = checked_file(source, rec["path"])
            if digest(data) != rec["SHA256"]:
                raise ValueError("Source changed during snapshot")
            target = repo / rec["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        # Explicit paths only. No recursive add, no -A, no weights or workspace scan.
        paths = [rec["path"] for rec in changed]
        for offset in range(0, len(paths), 32):
            git(repo, "add", "--", *paths[offset:offset + 32])
        staged = set(filter(None, git(repo, "diff", "--cached", "--name-only", "-z").stdout.decode().split("\0")))
        if staged != {r["path"] for r in changed}:
            raise ValueError("Staged files differ from reviewed plan")
        # Imported SHA-bound baseline bytes must not be reformatted. Check newly
        # authored/changed code; the exact initial baseline has separate hash review.
        whitespace_paths = [r["path"] for r in changed
                            if policy["files"][r["path"]].get("initial_SHA256") != r["SHA256"]]
        for name in whitespace_paths:
            git(repo, "-c", "core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol",
                "diff", "--cached", "--check", "--", name)
        for rec in changed:
            if git(repo, "show", ":" + rec["path"]).stdout != checked_file(source, rec["path"]):
                raise ValueError("Staged bytes differ from reviewed source")
        git(repo, "commit", "-m", message + "\n\nCode-Day: " + today)
        commits = audit_outgoing(repo, policy, base)
        git(repo, "push", "--set-upstream", "origin", "HEAD:refs/heads/" + policy["branch"])
        commit = git(repo, "rev-parse", "HEAD").stdout.decode().strip()
        actual = git(repo, "ls-remote", "origin", "refs/heads/" + policy["branch"]).stdout.decode().split()[0]
        if actual != commit:
            raise RuntimeError("Remote branch verification failed")
        return dict(status="pushed", day=today, branch=policy["branch"], commit=commit,
                    changed_files=len(changed), audited_outgoing_commits=commits)
    finally:
        if fd is not None:
            os.close(fd)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "publish"])
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--expected-plan-sha")
    parser.add_argument("--message", default="Maintain 910B evaluation source and configurations")
    args = parser.parse_args()
    if args.command == "publish" and not args.expected_plan_sha:
        parser.error("publish requires the SHA of a reviewed plan")
    result = plan_snapshot(args.source_root, args.repo_root, args.policy) if args.command == "plan" else publish(
        args.source_root, args.repo_root, args.policy, args.expected_plan_sha, args.message)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
