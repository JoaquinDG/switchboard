"""Prompt text, in one leaf module with no internal imports.

This exists so the audit prompt has a single owner. Previously the auditor
stamped a `[SWITCHBOARD_AUDIT]` marker onto every prompt purely so the offline
MockProvider could recognise an audit call — which meant a test double's
constant rode along on real API requests.

Now the mock keys off `AUDIT_PROMPT_HEADER`, a real line of the real prompt.
Both the auditor and the provider layer import from here, so they cannot drift
and nothing test-only reaches a vendor.
"""

from __future__ import annotations

# The first line of every audit prompt. MockProvider matches on it.
AUDIT_PROMPT_HEADER = "You are auditing another model's output."

AUDIT_PROMPT_TEMPLATE = (
    AUDIT_PROMPT_HEADER
    + """ Grade it strictly.

TASK TYPE: {task_type}
ORIGINAL PROMPT:
{prompt}

OUTPUT TO AUDIT:
{output}

Respond with ONLY a JSON object, no prose, no markdown fences:
{{"pass": true|false, "score": 0.0-1.0, "issues": ["..."]}}
Fail the output if it is incorrect, incomplete, unsafe, or ignores the prompt.
"""
)

# Escalation is a repair, not a blind re-roll. Handing the stronger model the
# auditor's findings costs a few dozen tokens and turns a second attempt into
# a targeted fix; re-sending the bare prompt throws away the one diagnostic
# the first attempt actually produced.
#
# The task's own prompt stays first and verbatim, and the audit findings are
# fenced off below it — they are model-generated text, and the boundary
# should be legible to the model reading it.
ESCALATION_RETRY_TEMPLATE = """{prompt}

---
A previous attempt at this task was audited and failed. The auditor reported:
{issues}

Produce a corrected response that addresses every issue above. Respond with
the corrected work itself, not a commentary on the previous attempt."""

# Cap on findings carried into a retry, so a verbose auditor cannot inflate
# the retry prompt without bound.
MAX_FEEDBACK_ISSUES = 10


def build_retry_prompt(prompt: str, issues: list[str]) -> str:
    """Fold audit findings into a retry prompt. No issues -> prompt unchanged."""
    if not issues:
        return prompt
    shown = [str(i) for i in issues[:MAX_FEEDBACK_ISSUES]]
    bullets = "\n".join(f"- {i}" for i in shown)
    remaining = len(issues) - len(shown)
    if remaining > 0:
        bullets += f"\n- (+{remaining} further issue(s) omitted)"
    return ESCALATION_RETRY_TEMPLATE.format(prompt=prompt, issues=bullets)
