# Security Policy

Switchboard is a library you run yourself, with your own keys. There is no
Switchboard service and no hosted endpoint. What it does have is a router
that decides which lab sees your prompt, which makes the interesting
vulnerabilities routing and key-handling bugs rather than server breaches.

## Reporting

Use GitHub's private vulnerability reporting:
**[Report a vulnerability](https://github.com/JoaquinDG/switchboard/security/advisories/new)**.
That opens a thread visible only to you and me.

If that is not available to you, email **joaquin.diaz@newryglobal.com** with
`[switchboard security]` in the subject line.

Please do not open a public issue for something you believe is exploitable.
For everything else a public issue is the right venue and is genuinely
welcome.

## What to expect

One maintainer, no security team, no bug bounty. What I can commit to:

- **Acknowledgement within 72 hours.** If you have not heard back by then,
  assume the message went astray and chase it.
- An assessment, with reasoning, within two weeks.
- Credit in the advisory and in the changelog, unless you would rather not
  be named.

I would much rather hear about something that turns out to be nothing than
not hear about it.

## In scope

- **Misrouting.** A task reaching a provider the policy did not select. A
  brokerage that sends your prompt to a lab you deliberately excluded is the
  worst bug this project can have, and it is a confidentiality bug, not a
  correctness one.
- **Key exposure.** Keys are read from the environment only: never from a
  file, never written to one, never logged, never traced. A key reaching a
  trace, a log line, a report, or a provider other than the one it belongs
  to is a serious bug. So is one provider's key being sent to another
  provider's endpoint.
- **Policy bypass.** Any route by which a configured constraint (excluded
  provider, spend ceiling, model floor) is not applied to a task that should
  have been subject to it.
- **Prompt injection through the audit path.** Cross-lab audits put one
  model's output in front of another model. A construction that steers the
  auditor or suppresses its finding is in scope.
- **Cost accounting that under-reports.** The guardrail is only meaningful
  if a session that burns extra calls reports a higher bill. Silent
  under-reporting defeats the control.
- **Artefacts that reach the network.** Traces and reports are meant to be
  self-contained. Anything that phones home when opened is a finding.

## Out of scope

- Models producing wrong, biased or unpleasant answers.
- Spend incurred by a configuration you chose, correctly reported.
- Vulnerabilities in the model providers themselves. Those belong to the
  provider.
- Anything that already assumes the attacker holds your API keys or has
  write access to your machine.

## Supported versions

`main` and the most recent release. There are no backports to older tags.
