# Issue tracker: GitHub

Issues and specs for this repo live in GitHub Issues for `huang2202/huang2202.github.io`. Use the `gh` CLI for tracker operations, running it from this repository so it infers the repo from the Git remote.

If `gh` is unavailable, use the connected GitHub tools for the same repository and operations. Keep the tracker and label conventions below unchanged.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body-file <markdown-file>`.
- **Read an issue**: `gh issue view <number> --comments`. Fetch labels with `gh issue view <number> --json number,title,body,labels,comments` when needed.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments`, with appropriate `--label` and `--state` filters. Paginate when a workflow needs the full queue.
- **Comment on an issue**: `gh issue comment <number> --body-file <markdown-file>`.
- **Apply or remove labels**: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`. Use the mapping in `triage-labels.md`.
- **Close an issue**: `gh issue close <number>`. If a closing explanation is needed, post it as a comment first.

For multiline issue bodies and comments, write the exact Markdown to a temporary file and pass it with `--body-file`. Preserve actual newlines.

## Pull requests as a triage surface

**PRs as a request surface: no.**

Set this flag to `yes` if the repo later treats external PRs as feature requests. The `triage` skill reads this flag. When enabled, use the `gh pr` equivalents to read, comment on, label, or close PRs, and inspect changes with `gh pr diff <number>`.

GitHub shares one number space across issues and PRs. If a bare `#42` is ambiguous, resolve it with `gh pr view 42` and fall back to `gh issue view 42`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`, also fetching labels when the workflow needs them.

## Wayfinding operations

Used by `wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue. Add the relation with `gh api --method POST repos/huang2202/huang2202.github.io/issues/<parent>/sub_issues -F sub_issue_id=<child-db-id>`. Where sub-issues are unavailable, add the child to a task list in the map body and put `Part of #<parent>` at the top of the child body. Labels: `wayfinder:<type>` (`research`, `prototype`, `grilling`, or `task`). Once claimed, assign the ticket to the driving developer.
- **Blocking**: use GitHub's native issue dependencies. Add an edge with `gh api --method POST repos/huang2202/huang2202.github.io/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`. Obtain the numeric database ID with `gh api repos/huang2202/huang2202.github.io/issues/<number> --jq .id`; it is distinct from the issue number and `node_id`. Where dependencies are unavailable, use a `Blocked by: #<number>, #<number>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children, scoped to its sub-issues or task list. Drop tickets with an assignee or an open blocker, using `issue_dependencies_summary.blocked_by` or the fallback `Blocked by` line. First in map order wins.
- **Claim**: `gh issue edit <number> --add-assignee @me`, before beginning ticket work.
- **Resolve**: post the answer with `gh issue comment <number> --body-file <markdown-file>`, close the ticket, then append a context pointer containing the gist and ticket link to the map's Decisions-so-far.
