# Loyal Wingman

Offload menial coding-agent tasks to a local LLM (via [LM Studio](https://lmstudio.ai)),
and teach it not to repeat mistakes.

Paid coding agents (Claude Code, etc.) are great at judgment and architecture,
wasteful for mechanical work: summarizing a log file, drafting a docstring,
formatting a changelog entry, rephrasing text that's already fully specified.
Loyal Wingman gives an agent (or you) a way to hand that work to a local
model instead, review the result, and correct it when it's wrong -- without
spending metered tokens on it.

## How it works

1. **`run`** -- send a task to whatever model is loaded in LM Studio.
   Auto-starts the LM Studio server and auto-loads a model if neither is
   already up.
2. You (or the calling agent) **check the result**.
3. If it's wrong, **`teach`** the correction. It's appended to a lessons
   file and automatically prepended to future prompts as in-context
   guidance -- so the same mistake gets caught before it repeats.

This is *not* fine-tuning. Most locally-served models are static weights
with no online-learning API. Teaching here means in-context correction, not
a weight update -- cheap, immediate, and good enough for catching repeated
mistakes without a training pipeline.

## Install

```bash
pip install -e .
```

Requires [LM Studio](https://lmstudio.ai) installed, with at least one model
downloaded (`lms get <model>` from LM Studio's CLI, or via the desktop app).

## Usage

```bash
# Run a task
loyal-wingman run "Summarize this in 2 sentences: <text>"

# Large input via file (avoids shell quoting/length issues)
loyal-wingman run --file big_log_excerpt.txt \
  --system "Summarize the errors in this log. Be terse."

# Piped input
some_command | loyal-wingman run --system "Reformat as a bullet list."

# Tag a task by category so relevant lessons get applied
loyal-wingman run "..." --category changelog

# After reviewing output and finding a mistake, record the correction
loyal-wingman teach --category changelog \
  --issue "Wrote a plain sentence instead of the house style" \
  --fix "Format as: **Title** (\`file.py\`): terse technical description"

# Review what's been taught so far
loyal-wingman lessons
loyal-wingman lessons --category changelog
```

Lessons tagged `general` (the default when `--category` is omitted on
`teach`) apply to every `run` regardless of category. Namespace
project-specific conventions with their own category
(e.g. `myproject-changelog`) so a correction from one project doesn't leak
into an unrelated one.

## Model selection

`run` picks a model in this order:

1. `--model <id>` if given
2. Whatever's already loaded in LM Studio
3. `LOYAL_WINGMAN_MODEL` env var
4. The sole downloaded model, if exactly one is available

If none of these resolve unambiguously, it errors with the list of
downloaded models rather than guessing.

## Configuration

- `LOYAL_WINGMAN_MODEL` -- default model id when nothing else applies
- `LOYAL_WINGMAN_HOME` -- where the lessons file lives (default: `~/.loyal-wingman/`)

## Design notes

- Calls LM Studio's own OpenAI-compatible endpoint directly
  (`http://localhost:1234/v1`) via the Python standard library only --
  no extra dependencies, no third-party proxy.
- Reasoning models are commonly served with a chat template that
  auto-prepends the `<think>` opening tag before generation starts (so it
  never appears in the response), but the model's own `</think>` closing
  tag does. `run` strips everything through that marker so callers get the
  final answer, not the chain-of-thought -- a no-op for models that don't
  use this convention.
- Windows console output is reconfigured to UTF-8 with `errors="replace"`,
  since local-model output routinely contains characters (em-dashes,
  checkmarks) outside legacy code pages.

## License

MIT
