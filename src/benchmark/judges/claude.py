"""Claude Code CLI judge: `claude -p` under the user's own Claude subscription login.

Official docs: https://code.claude.com/docs/en/headless (non-interactive `claude -p`).
`--bare` is NOT used because bare mode never reads the subscription OAuth login
(it requires ANTHROPIC_API_KEY, i.e. separately billed API usage). Instead the
harness is minimised explicitly:
  --system-prompt   replaces Claude Code's default system prompt with the evaluator's
                    own system message (Q-CARE) or NEUTRAL_SYSTEM
  --tools ""        no tools;  --strict-mcp-config: no MCP servers
  --setting-sources "" no user/project settings or hooks;  --disable-slash-commands
  --no-session-persistence; empty temporary working directory
Model id actually served is read from the JSON `modelUsage` field and logged.
`total_cost_usd` in the CLI output is a client-side estimate; on a subscription
login the usage counts against plan limits and is not billed per call.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Optional

from .codex import NEUTRAL_SYSTEM, CodexJudge

CLAUDE_HARNESS = "claude-judge-v1"


class ClaudeJudge(CodexJudge):
    backend = "claude-cli"
    harness = CLAUDE_HARNESS

    def __init__(self, model: str = "sonnet", effort: Optional[str] = None, claude_bin: Optional[str] = None, **kw):
        super().__init__(model=model, reasoning_effort=effort or "default", codex_bin=None, **kw)
        self.claude_bin = claude_bin or shutil.which("claude")
        self.effort = effort

    def _bin_available(self) -> bool:
        return bool(self.claude_bin)

    def describe(self):
        if self._version is None and self.claude_bin:
            try:
                self._version = subprocess.run([self.claude_bin, "--version"], capture_output=True, text=True,
                                               timeout=30).stdout.strip()
            except Exception:  # noqa: BLE001
                self._version = "unknown"
        return {"backend": self.backend, "model": self.model, "effort": self.effort or "cli default",
                "cli_version": self._version, "harness": CLAUDE_HARNESS, "replicate": self.replicate,
                "temperature": "not controllable via claude -p",
                "model_identity": "served model reported per uncached call (modelUsage -> model_reported_by_cli)",
                **({"cache_replay_only": True,
                    "cli_version_note": "no new calls allowed; responses come from cache records, each of which "
                                        "stores the CLI version that produced it"} if (self.max_new_calls == 0 or self.offline) else {})}

    def _invoke(self, prompt, sys_text, output_schema):
        cmd = [self.claude_bin, "-p", "--output-format", "json", "--model", self.model,
               "--system-prompt", sys_text or NEUTRAL_SYSTEM, "--tools", "", "--strict-mcp-config",
               "--setting-sources", "", "--disable-slash-commands", "--no-session-persistence"]
        if self.effort:
            cmd += ["--effort", self.effort]
        if output_schema:
            cmd += ["--json-schema", json.dumps(output_schema)]
        try:
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout_sec,
                               cwd=self._workdir)
        except subprocess.TimeoutExpired:
            return "", {}, None, f"timeout after {self.timeout_sec}s"
        try:
            out = json.loads(p.stdout)
        except json.JSONDecodeError:
            return "", {}, None, f"rc={p.returncode} non-json stdout={p.stdout[-300:]} stderr={p.stderr[-300:]}"
        usage = dict(out.get("usage") or {})
        model_usage = out.get("modelUsage") or {}
        usage["model_usage"] = model_usage
        usage["cost_estimate_usd_client_side"] = out.get("total_cost_usd")
        models = sorted(model_usage)
        text = (out.get("result") or "").strip()
        if output_schema and out.get("structured_output") is not None:
            text = json.dumps(out["structured_output"])
        if p.returncode != 0 or out.get("is_error") or not text:
            # Keep the CLI's full result text (it ends with the provider's refusal category and request id,
            # e.g. "Details: `[reasoning_extraction]`") and the stop/safety fields needed to diagnose it.
            return "", usage, ",".join(models), (
                f"rc={p.returncode} is_error={out.get('is_error')} stop_reason={out.get('stop_reason')} "
                f"terminal_reason={out.get('terminal_reason')} safety_stops={out.get('safety_stops')} "
                f"api_error_status={out.get('api_error_status')} result={str(out.get('result'))[:4000]}")
        return text, usage, ",".join(models), None
