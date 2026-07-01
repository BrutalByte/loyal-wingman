# Loyal Wingman

[![Tests](https://github.com/BrutalByte/loyal-wingman/actions/workflows/tests.yml/badge.svg)](https://github.com/BrutalByte/loyal-wingman/actions/workflows/tests.yml)

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
   file. When you `run` a future task, the most semantically similar past
   lessons are automatically retrieved and prepended as in-context guidance
   -- so a relevant correction surfaces even if you don't remember the exact
   category you taught it under.

This is *not* fine-tuning. Most locally-served models are static weights
with no online-learning API. Teaching here means in-context correction, not
a weight update -- cheap, immediate, and good enough for catching repeated
mistakes without a training pipeline.

## Semantic lesson retrieval

Each lesson's `issue` text is embedded (via a local embedding model, e.g.
`nomic-embed-text`) when taught. Each `run`'s prompt is embedded the same
way, and lessons above a similarity floor are retrieved, ranked by
closeness, and prepended to the system prompt -- regardless of category tag.
`--category` still works as an optional pre-filter if you want to guarantee
only a specific bucket of lessons is considered.

If no embedding model is downloaded, or the embedding call fails for any
reason, this degrades gracefully to "most recent N lessons in category" --
semantic retrieval is an enhancement, never a hard requirement to run a
task. Lessons taught before an embedding model was available (or before
this feature existed) get embedded lazily on the next `run` that has one
loaded.

## Tests

Unit tests cover the lessons storage/retrieval logic (`tests/test_retrieval.py`)
with no network or LM Studio server required -- `_embed` is mocked. Uses
only the standard library (`unittest.mock`), so no test dependency to install:

```bash
python -m unittest discover -s tests
```

Also runs under pytest if you have it installed (`pytest tests/`).

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

# Optional: restrict lesson matching to a category before ranking by similarity
loyal-wingman run "..." --category changelog

# Tune retrieval (defaults: top 5 lessons, similarity floor 0.58)
loyal-wingman run "..." --top-lessons 3 --min-similarity 0.65

# Skip lesson retrieval entirely (faster; no embedding model needed)
loyal-wingman run "..." --no-lessons

# After reviewing output and finding a mistake, record the correction
loyal-wingman teach --category changelog \
  --issue "Wrote a plain sentence instead of the house style" \
  --fix "Format as: **Title** (\`file.py\`): terse technical description"

# Review what's been taught so far (shows embedding status per lesson)
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
downloaded models rather than guessing. Embedding model selection follows
the same order via `--embed-model` / `LOYAL_WINGMAN_EMBED_MODEL`, except
that an ambiguous or missing embedding model degrades to non-semantic
lesson retrieval instead of erroring.

## Configuration

- `LOYAL_WINGMAN_MODEL` -- default model id when nothing else applies
- `LOYAL_WINGMAN_EMBED_MODEL` -- default embedding model id for lesson retrieval
- `LOYAL_WINGMAN_HOME` -- where the lessons file lives (default: `~/.loyal-wingman/`)

## Known limitation: GPU offload when using two models

`run` loads the embedding model before the main model so that, on a cold
start, the main model's automatic GPU offload accounts for both models'
footprint. This does **not** protect against a main model that was already
loaded (by something else, or by a previous `loyal-wingman run`) before the
embedding model loads for the first time: LM Studio can rebalance VRAM
allocation when a second model loads, silently evicting part of an
already-resident model to CPU. Observed effect: a fully GPU-offloaded 21GB
model dropped from ~1.5s to 90+s per response. If generation suddenly gets
much slower after lessons start getting used, reload the main model
explicitly:

```bash
lms unload --all
lms load <model> --gpu max
```

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
