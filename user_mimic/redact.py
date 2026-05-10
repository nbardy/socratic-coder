from __future__ import annotations

import re

SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}"),
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(
        r"-----BEGIN (OPENSSH |RSA )?PRIVATE KEY-----[\s\S]*?-----END",
    ),
    re.compile(r"Bearer [a-zA-Z0-9_.-]{20,}"),
]

REDACTED = "[REDACTED_SECRET]"
TOOL_RESULT_KEEP_BELOW = 200


def redact_secrets(text: str) -> str:
    out = text
    for pat in SECRET_PATTERNS:
        out = pat.sub(REDACTED, out)
    return out


def render_tool_result_body(body: str, is_error: bool = False) -> str:
    if len(body) < TOOL_RESULT_KEEP_BELOW:
        return redact_secrets(body)
    return f"[tool_result: {len(body)} bytes]"
