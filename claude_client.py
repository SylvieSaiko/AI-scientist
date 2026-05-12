"""
Drop-in Claude caller using the local `claude` CLI (Claude Code OAuth session).
No ANTHROPIC_API_KEY required.
"""

import subprocess, shutil, sys, os

_CLAUDE_BIN = shutil.which("claude")
if not _CLAUDE_BIN:
    sys.exit("ERROR: `claude` CLI not found. Is Claude Code installed?")


def claude_call(prompt: str, model: str = "claude-sonnet-4-6",
                max_tokens: int = 4096, system: str = None) -> str:
    cmd = [_CLAUDE_BIN, "--print", "--output-format", "text", "--model", model]
    if system:
        prompt = f"<system>{system}</system>\n\n{prompt}"
    result = subprocess.run(
        cmd, input=prompt,
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    output = result.stdout.strip()
    # CLI returns non-zero when it uses tool calls; treat as error only if no output
    if not output:
        err = result.stderr.strip() or "(no output)"
        raise RuntimeError(f"claude CLI returned no output:\n{err[:1000]}")
    # Strip a leading "Error: ..." line that sometimes appears before real content
    lines = output.splitlines()
    if lines and lines[0].startswith("Error:"):
        output = "\n".join(lines[1:]).strip()
    if not output:
        raise RuntimeError(f"claude CLI error: {lines[0] if lines else '(empty)'}")
    return output
