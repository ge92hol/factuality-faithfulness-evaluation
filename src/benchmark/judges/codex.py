"""Codex CLI judge: runs `codex exec` non-interactively under the user's ChatGPT login.

Officially documented path: https://learn.chatgpt.com/docs/non-interactive-mode
(`codex exec` reuses saved CLI auth; `--json` emits a `turn.completed` event with
token usage). Usage counts against the ChatGPT plan allowance, not API billing.

Harness minimisation (ADAPTATION, documented in README):
  * `--ignore-user-config`, `--ephemeral`, `--sandbox read-only`, empty working dir
  * agent tools / plugins / browser / apps features disabled
  * `base_instructions` replaced by the evaluator's own system prompt when it has
    one (Q-CARE), otherwise by NEUTRAL_SYSTEM below
  * no temperature control is exposed by Codex -> outputs are not guaranteed
    deterministic; every response is cached by content hash so reported runs are
    exactly replayable, and replicates are requested explicitly (T9).
A residual ~8-9k-token Codex harness context is still sent with each call.
"""

from __future__ import annotations

import contextvars
import glob
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Judge, JudgeError, JudgeResponse

NEUTRAL_SYSTEM = (
    "You are a helpful assistant. Answer the user's request directly with text only. "
    "Do not use tools, do not run commands, do not ask questions."
)

DISABLED_FEATURES = [
    "shell_tool", "unified_exec", "apps", "plugins", "multi_agent", "browser_use",
    "browser_use_external", "computer_use", "image_generation", "view_image",
    "skill_search", "tool_suggest", "goals", "sleep_tool", "hooks", "in_app_browser",
    "plugin_discovery_prefetch", "remote_plugin", "skill_mcp_dependency_install",
]
FALSE_CONFIG = [
    "include_apps_instructions", "include_apps_usage_instructions",
    "include_collaboration_mode_instructions", "include_environment_context",
    "include_permissions_instructions", "include_plugin_usage_instructions",
    "include_skills_usage_instructions",
]
HARNESS_VERSION = "codex-judge-v1"
QUOTA_MARKERS = ("usage limit", "rate limit", "quota", "429", "too many requests")


def find_codex_bin() -> Optional[str]:
    if os.environ.get("CODEX_BIN"):
        return os.environ["CODEX_BIN"]
    if shutil.which("codex"):
        return shutil.which("codex")
    pats = [
        os.path.expanduser("~/.vscode-server/extensions/openai.chatgpt-*/bin/linux-x86_64/codex"),
        os.path.expanduser("~/.vscode/extensions/openai.chatgpt-*/bin/*/codex"),
    ]
    hits = sorted(h for p in pats for h in glob.glob(p))
    return hits[-1] if hits else None


def codex_status(codex_bin: Optional[str] = None) -> Dict[str, Any]:
    """Report binary/version/login without reading secrets."""
    b = codex_bin or find_codex_bin()
    out: Dict[str, Any] = {"codex_bin": b, "available": False}
    if not b:
        out["reason"] = "codex binary not found (set CODEX_BIN)"
        return out
    try:
        out["version"] = subprocess.run([b, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
        st = subprocess.run([b, "login", "status"], capture_output=True, text=True, timeout=30)
        out["login_status"] = (st.stdout + st.stderr).strip()
        out["available"] = "Logged in" in out["login_status"]
    except Exception as e:  # noqa: BLE001
        out["reason"] = repr(e)
    return out


class CodexJudge(Judge):
    backend = "codex-cli"

    def __init__(
        self,
        model: str = "gpt-6-luna",
        reasoning_effort: str = "low",
        codex_bin: Optional[str] = None,
        cache_dir: str = "cache/judge",
        call_log: Optional[str] = None,
        max_workers: int = 4,
        timeout_sec: int = 300,
        max_retries: int = 2,
        replicate: int = 0,
        offline: bool = False,
        max_new_calls: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.codex_bin = codex_bin or find_codex_bin()
        self.cache_dir = Path(cache_dir).resolve()  # absolute: official code may chdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.call_log = Path(call_log).resolve() if call_log else None
        if self.call_log:
            self.call_log.parent.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.replicate = replicate
        self.offline = offline  # cache-only replay: a miss raises instead of calling
        self.max_new_calls = max_new_calls  # hard cap on uncached calls (quota guard)
        self._new_calls = 0
        self._lock = threading.Lock()
        self._workdir = tempfile.mkdtemp(prefix="codex_judge_cwd_")
        self._version = None

    harness = HARNESS_VERSION

    def _bin_available(self) -> bool:
        return bool(self.codex_bin)

    # ---------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        if self._version is None and self.codex_bin:
            try:
                self._version = subprocess.run([self.codex_bin, "--version"], capture_output=True,
                                               text=True, timeout=30).stdout.strip()
            except Exception:  # noqa: BLE001
                self._version = "unknown"
        return {
            "backend": self.backend, "model": self.model, "reasoning_effort": self.reasoning_effort,
            "cli_version": self._version, "harness": HARNESS_VERSION, "replicate": self.replicate,
            "temperature": "not controllable via codex exec",
            "model_identity": "requested identifier only; codex exec does not report the served model (UNVERIFIED)",
            **({"cache_replay_only": True,
                "cli_version_note": "no new calls allowed; responses come from cache records, each of which stores "
                                    "the CLI version that produced it"} if (self.max_new_calls == 0 or self.offline) else {}),
        }

    def _key(self, prompt: str, system: Optional[str], schema: Optional[dict]) -> str:
        blob = json.dumps({
            "h": self.harness, "m": self.model, "e": self.reasoning_effort,
            "s": system or NEUTRAL_SYSTEM, "p": prompt, "o": schema, "r": self.replicate,
            **({"b": self.backend} if self.backend != "codex-cli" else {}),  # codex keys unchanged
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _log(self, rec: Dict[str, Any]) -> None:
        if self.call_log:
            with self._lock, open(self.call_log, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def _cmd(self, system: str, out_file: str, schema_file: Optional[str]) -> List[str]:
        cmd = [self.codex_bin, "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
               "--sandbox", "read-only", "-C", self._workdir, "--color", "never",
               "-m", self.model, "-c", f"model_reasoning_effort={self.reasoning_effort}",
               "-c", f"base_instructions={json.dumps(system)}",
               "-c", "web_search=disabled", "-c", "project_doc_max_bytes=0", "--json", "-o", out_file]
        for f in DISABLED_FEATURES:
            cmd += ["--disable", f]
        for k in FALSE_CONFIG:
            cmd += ["-c", f"{k}=false"]
        if schema_file:
            cmd += ["--output-schema", schema_file]
        cmd.append("-")
        return cmd

    def _key_lock(self, key: str) -> threading.Lock:
        with self._lock:
            if not hasattr(self, "_inflight"):
                self._inflight = {}
            return self._inflight.setdefault(key, threading.Lock())

    def _complete(self, prompt, system=None, tag="", output_schema=None) -> JudgeResponse:
        """Single-flight per cache key: identical prompts issued concurrently (e.g. the same
        reference extracted once per candidate in a RAGChecker batch) share ONE judge call,
        as they would with a deterministic temperature-0 backend."""
        key = self._key(prompt, system, output_schema)
        with self._key_lock(key):
            return self._complete_locked(key, prompt, system, tag, output_schema)

    def _complete_locked(self, key, prompt, system, tag, output_schema) -> JudgeResponse:
        cpath = self.cache_dir / f"{key}.json"
        if cpath.exists():
            rec = json.loads(cpath.read_text())
            r = JudgeResponse(rec["response"], rec["model"], self.backend, True, 0.0,
                              rec.get("usage", {}), rec["call_id"])
            self._log({"event": "cache_hit", "key": key, "call_id": rec["call_id"], "tag": tag})
            return r
        if self.offline:
            self._log({"event": "error", "key": key, "tag": tag, "error": "offline cache miss"})
            raise JudgeError(f"offline replay: no cached judgment for key {key[:12]} (tag={tag})")
        if not self._bin_available():
            raise JudgeError(f"{self.backend} binary not found")
        with self._lock:
            if self.max_new_calls is not None and self._new_calls >= self.max_new_calls:
                raise JudgeError(f"max_new_calls cap ({self.max_new_calls}) reached")
            self._new_calls += 1

        sys_text = system or NEUTRAL_SYSTEM
        last_err = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            text, usage, model_seen, err = self._invoke(prompt, sys_text, output_schema)
            latency = time.time() - t0
            if err:
                last_err = err
                if any(m in err.lower() for m in QUOTA_MARKERS):
                    break  # do not hammer a quota limit
                continue
            call_id = str(uuid.uuid4())
            rec = {"call_id": call_id, "key": key, "backend": self.backend, "model": self.model,
                   "reasoning_effort": self.reasoning_effort, "cli_version": self.describe().get("cli_version"),
                   "harness": self.harness, "replicate": self.replicate, "system": sys_text,
                   "prompt": prompt, "output_schema": output_schema, "response": text, "usage": usage,
                   "latency_sec": round(latency, 3), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "tag": tag, "model_reported_by_cli": model_seen}
            cpath.write_text(json.dumps(rec, indent=1))
            self._log({"event": "call", **{k: rec[k] for k in ("call_id", "key", "model", "usage", "latency_sec", "tag")},
                       "model_reported_by_cli": model_seen})
            return JudgeResponse(text, self.model, self.backend, False, latency, usage, call_id)
        self._log({"event": "error", "key": key, "tag": tag, "error": last_err})
        raise JudgeError(last_err or f"unknown {self.backend} failure")

    def _invoke(self, prompt, sys_text, output_schema):
        """One CLI call -> (text, usage, model_reported, error_or_None)."""
        with tempfile.TemporaryDirectory(prefix="cj_") as td:
            out_file = os.path.join(td, "last.txt")
            schema_file = None
            if output_schema:
                schema_file = os.path.join(td, "schema.json")
                Path(schema_file).write_text(json.dumps(output_schema))
            try:
                p = subprocess.run(self._cmd(sys_text, out_file, schema_file), input=prompt,
                                   capture_output=True, text=True, timeout=self.timeout_sec)
            except subprocess.TimeoutExpired:
                return "", {}, None, f"timeout after {self.timeout_sec}s"
            usage, errors, model_seen = {}, [], None
            for line in p.stdout.splitlines():
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = ev.get("type", "")
                if t == "turn.completed":
                    usage = ev.get("usage", {})
                if t in ("error", "turn.failed"):
                    errors.append(json.dumps(ev)[:500])
                model_seen = model_seen or ev.get("model")
            text = Path(out_file).read_text().strip() if os.path.exists(out_file) else ""
            if p.returncode != 0 or errors or not text:
                return "", usage, model_seen, f"rc={p.returncode} errors={errors} stderr={p.stderr[-500:]}"
            return text, usage, model_seen, None

    def batch(self, prompts: List[str], tag: str = "") -> List[JudgeResponse]:
        # Each prompt runs in a copy of the caller's context, so the caller's open ledgers
        # (and only those) record these calls; a Context cannot be entered by two threads at once.
        ctxs = [contextvars.copy_context() for _ in prompts]
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda p, c: c.run(self.complete, p, tag=tag), prompts, ctxs))
