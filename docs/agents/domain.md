# Domain Docs

How engineering skills consume this repo's terminology and architecture decisions.

## Before exploring, read these

- Root `GLOSSARY.md`, when present.
- ADRs in `docs/adr/` that touch the area being explored.

If these files do not exist, proceed silently. The `domain-modeling` skill creates them lazily when terms or decisions are resolved.

## File structure

This repo uses a **single-context** layout. The website and its theme package share one glossary and one ADR directory.

```text
/
├── GLOSSARY.md
├── docs/
│   ├── agents/
│   └── adr/
│       └── 0001-<decision-slug>.md
├── src/
└── packages/
```

## Use the glossary's vocabulary

When naming a domain concept in an issue, proposal, hypothesis, or test, use the term defined in `GLOSSARY.md`.

If a needed concept is missing, reconsider whether it matches the project's language. Record real vocabulary gaps for `domain-modeling`.

## Flag ADR conflicts

If a proposal contradicts an existing ADR, name the ADR and explain why the decision should be reopened before replacing it.
