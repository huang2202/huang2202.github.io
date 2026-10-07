# Repository instructions

## Luna worker

实施/审阅子任务显式使用 `luna_worker`，不依赖父代理继承模型。

每个新 subagent 必须明确指定：

- `agent_type = luna_worker`
- `model = gpt-6-luna`
- `reasoning_effort = max`
- `fork_turns = none`

上述要求也适用于子代理创建的后续子代理。因为不继承对话，分派时必须提供自足的任务说明、工作目录、相关文件和必要约束。

如果当前创建接口不支持上述必需参数，说明限制，不得静默省略参数、依赖继承值或替换模型。

## Agent skills

### Issue tracker

Issues and specs live in this repo's GitHub Issues. Before reading or writing tickets, read `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default triage labels. Before triaging tickets, read `docs/agents/triage-labels.md`.

### Domain docs

Use a single-context layout: root `GLOSSARY.md` and `docs/adr/`. Before exploring the codebase, read `docs/agents/domain.md`.

## Website design

Before changing public pages or note visibility, read `docs/design/academic-homepage.md` for the agreed content, identity, and publication scope.
