# Agent notes for this fork

This repository is a fork of [54yyyu/zotero-mcp](https://github.com/54yyyu/zotero-mcp)
with a few local changes: the `check-missing-fulltext` command
(`src/zotero_mcp/check_missing_fulltext.py`) and a CI reduced to
`.github/workflows/tests.yml`.

## Syncing with upstream

- Merge `upstream/main` into a branch; never rebase onto it.
- `docker.yml` and `release.yml` stay deleted (modify/delete conflict on every sync).
- `cli.py`: keep the `check-missing-fulltext` subparser next to whatever upstream added.
- `check-missing-fulltext` must stay listed in `HELP_GROUPS` (`src/zotero_mcp/cli_help.py`),
  or `tests/test_cli_help.py` fails.
- An upstream sync PR is merged with a merge commit, never squashed: squashing
  drops upstream as a parent, so the next sync starts from the old merge base
  and replays every conflict.

## Code review with Coderabbit

Coderabbit does not review this repository automatically (it has fewer than 10 stars), so you must trigger every review yourself: after opening the PR, add a comment containing exactly `@coderabbitai review`. Retrigger the review after every push, documentation included, since every push dismisses the approval (edits to the PR title or body are not pushes and need none). Coderabbit reviews at most once per hour; a trigger inside that window is not run: Coderabbit edits its acknowledgement of the trigger to "Action not completed: Review rate limited", and its summary comment at the top of the PR into a "Review limit reached" notice naming the wait. The trigger is dropped, not queued.

Fix no-brainer feedback directly; discuss anything that is a judgement call with the requester instead of deciding it on your own. Answer every Coderabbit thread whose suggestion was declined: Coderabbit approves only once all its threads are resolved, and lists any that are not in an "Automatic approval withheld" notice in its summary comment. Resolve a declined thread only after the requester has agreed to decline it.

Enable auto-merge on the PR right after opening it: squash for ordinary PRs, a merge commit for an upstream sync (see above). The ruleset on main requires an approving review and green `test (3.10)`, `test (3.11)` and `test (3.12)` checks (`tests.yml`), and dismisses the approval on every push. Coderabbit's approval is that review, so the PR merges by itself once the review round is done; nobody merges by hand.

Do not rely on PR events reaching you. After every trigger, schedule a check-in with send_later: 10 minutes after the trigger, or a minute past the wait a rate-limit notice names. On the check-in, read the PR: merged means done; no reaction from Coderabbit to the trigger means it was lost, so trigger once more; a rate-limit notice means trigger again once it has run out; requested changes mean fixing and discussing as above; a failed test job means finding its cause and fixing it, or reporting the blocker to the requester. Schedule a new check-in after each step until the PR has merged or waits on the requester.
