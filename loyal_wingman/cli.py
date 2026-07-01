"""
Loyal Wingman -- a CLI for offloading menial coding-agent tasks to a local
LM Studio model, with a persistent "lessons" store so corrections don't
have to be repeated.

Role split: the calling agent (or you) formulates the request, reviews the
result, and -- if it's wrong -- teaches the correction. The local model does
the mechanical work. This is intentionally not weight-level fine-tuning:
most locally-served models are static weights with no built-in
online-learning API. "Teaching" here means appending to a lessons file and
prepending matching lessons to future prompts as in-context guidance --
cheap, immediate, and good enough for catching repeated mistakes without a
training pipeline.

Lessons are retrieved by semantic similarity, not exact category match: each
lesson's "issue" text is embedded (via a local embedding model, e.g.
nomic-embed-text) when taught, the current prompt is embedded when running a
task, and the most similar past lessons (above a similarity floor) are
prepended -- so a relevant correction surfaces even if you don't remember
the exact category tag you used when teaching it. If no embedding model is
available, this degrades gracefully to "most recent N lessons in category".

`run` auto-starts LM Studio's server and auto-loads a model if neither is
already up -- no manual `lms server start` / `lms load` step required.

Usage:
    # Run a task (auto-starts LM Studio + loads a model if needed)
    loyal-wingman run "prompt text"
    loyal-wingman run --file big_input.txt --system "Summarize concisely." --category log-summary
    some_command | loyal-wingman run --system "..."

    # Record a correction after reviewing output
    loyal-wingman teach --category changelog \\
        --issue "What it got wrong" \\
        --fix "What it should have done instead"

    # Review accumulated lessons
    loyal-wingman lessons
    loyal-wingman lessons --category changelog
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Local-model output routinely contains characters (em-dashes, checkmarks,
# box-drawing) outside legacy Windows console code pages (cp1252/cp437),
# which would otherwise raise UnicodeEncodeError on print/write.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

LESSONS_PATH = Path(
    os.environ.get("LOYAL_WINGMAN_HOME", Path.home() / ".loyal-wingman")
) / "lessons.jsonl"
DEFAULT_TTL_SECONDS = 1800
DEFAULT_TOP_LESSONS = 5
# Empirically calibrated against nomic-embed-text-v1.5 on a handful of real
# vs. unrelated task prompts: genuinely related short technical phrases
# scored ~0.62-0.64 cosine similarity, unrelated ones ~0.50-0.56. This is a
# starting point, not a proven constant -- tune with --min-similarity as
# your own lesson corpus grows and you observe false positives/negatives.
DEFAULT_MIN_SIMILARITY = 0.58
# Embedding text is truncated to this many characters before sending, so a
# large --file input doesn't blow past the embedding model's (typically
# much smaller than the chat model's) context window.
MAX_EMBED_CHARS = 2000
LMS_FALLBACK_PATHS = [
    Path.home() / ".lmstudio" / "bin" / "lms",
    Path.home() / ".lmstudio" / "bin" / "lms.exe",
]


# ── LM Studio process management ─────────────────────────────────────────

def _lms_bin() -> str:
    found = shutil.which("lms")
    if found:
        return found
    for candidate in LMS_FALLBACK_PATHS:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "Could not find the 'lms' CLI on PATH or at ~/.lmstudio/bin/lms. "
        "Install LM Studio: https://lmstudio.ai"
    )


def _server_reachable(base_url: str) -> bool:
    try:
        urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=3)
        return True
    except Exception:
        return False


def _ensure_server(base_url: str, startup_timeout: float = 60.0) -> None:
    if _server_reachable(base_url):
        return
    print("LM Studio server not running -- starting it...", file=sys.stderr)
    subprocess.run(
        [_lms_bin(), "server", "start"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
    )
    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if _server_reachable(base_url):
            return
        time.sleep(1)
    raise ConnectionError(f"LM Studio server did not come up within {startup_timeout:.0f}s")


def _loaded_models() -> list:
    try:
        result = subprocess.run(
            [_lms_bin(), "ps", "--json"],
            capture_output=True, text=True, timeout=15,
        )
        return json.loads(result.stdout or "[]")
    except Exception:
        return []


def _available_models(model_type: str) -> list:
    """Locally downloaded models of the given lms type ("llm" or "embedding")."""
    try:
        result = subprocess.run(
            [_lms_bin(), "ls", "--json"],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(result.stdout or "[]")
        return [m for m in data if m.get("type") == model_type]
    except Exception:
        return []


def _available_llms() -> list:
    """Locally downloaded LLM model ids (excludes embedding models)."""
    return [m["modelKey"] for m in _available_models("llm")]


def _ensure_model(model: str, ttl: int, load_timeout: float = 180.0) -> str:
    # Must filter to type == "llm" -- _loaded_models() also returns any
    # loaded embedding model, and an unfiltered list here previously caused
    # a loaded embedding model to be mistaken for "the loaded LLM" and sent
    # a chat completion request (which LM Studio correctly rejected).
    loaded_ids = [m.get("identifier") for m in _loaded_models() if m.get("type") == "llm"]
    if model:
        if model in loaded_ids:
            return model
        target = model
    elif loaded_ids:
        return loaded_ids[0]
    else:
        target = os.environ.get("LOYAL_WINGMAN_MODEL", "")
        if not target:
            available = _available_llms()
            if len(available) == 1:
                target = available[0]  # unambiguous -- just use it
            elif available:
                raise RuntimeError(
                    "No model is loaded and more than one is downloaded -- pass "
                    "--model <id> or set LOYAL_WINGMAN_MODEL.\nAvailable: " + ", ".join(available)
                )
            else:
                raise RuntimeError(
                    "No model is loaded and none are downloaded. Download one first, "
                    "e.g.: lms get <model-name> (see https://lmstudio.ai/models)"
                )

    print(f"Loading {target} into LM Studio (can take up to a minute)...", file=sys.stderr)
    # --gpu max forces full GPU offload. Without it, LM Studio's automatic
    # offload heuristic can under-offload a model (observed: a 21GB model on
    # a 32GB GPU left partially on CPU, dropping generation from ~1.5s to
    # 45+s for a trivial prompt -- reported 100% GPU "utilization" the whole
    # time, but at a fraction of the card's actual power draw, since most of
    # the work was happening off-GPU). This only affects models THIS call
    # loads; a model left loaded by something else (or a prior loyal-wingman
    # run before this fix) is trusted as-is and not reloaded.
    proc = subprocess.run(
        [_lms_bin(), "load", target, "--gpu", "max", "--ttl", str(ttl)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=load_timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to load model {target!r}: {proc.stderr.strip()}")
    return target


def _ensure_embedding_model(model: str, ttl: int, load_timeout: float = 60.0) -> str:
    """
    Best-effort: returns an embedding model id ready to use, or "" if none
    is available/loadable. Semantic lesson retrieval is an enhancement, not
    a hard requirement, so callers should degrade gracefully on "" rather
    than treat it as a fatal error the way _ensure_model's failures are.
    """
    for m in _loaded_models():
        if m.get("type") == "embedding":
            if not model or model == m.get("identifier"):
                return m.get("identifier")

    target = model or os.environ.get("LOYAL_WINGMAN_EMBED_MODEL", "")
    available = _available_models("embedding")
    if not target:
        if len(available) == 1:
            target = available[0]["modelKey"]
        else:
            return ""  # none or ambiguous -- caller falls back to recency-based selection

    # lms load matches fuzzily against the download path, not the modelKey
    # used by the API -- an id like "text-embedding-nomic-embed-text-v1.5"
    # can fail to resolve and drop into an interactive picker. Use the
    # known path + --exact instead, which resolves deterministically.
    entry = next((m for m in available if m["modelKey"] == target), None)
    load_target = entry["path"] if entry else target
    try:
        proc = subprocess.run(
            [_lms_bin(), "load", load_target, "--exact", "--ttl", str(ttl)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=load_timeout,
        )
    except Exception:
        return ""
    return target if proc.returncode == 0 else ""


def _embed(base_url: str, model: str, text: str, timeout: float = 30.0):
    """Returns an embedding vector (list[float]), or None on any failure."""
    payload = json.dumps({"model": model, "input": text[:MAX_EMBED_CHARS]}).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/embeddings", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["data"][0]["embedding"]
    except Exception:
        return None


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ── Lessons store ─────────────────────────────────────────────────────────

def _load_lessons() -> list:
    if not LESSONS_PATH.exists():
        return []
    lessons = []
    with open(LESSONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lessons.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return lessons


def _save_lessons(lessons: list) -> None:
    LESSONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LESSONS_PATH, "w", encoding="utf-8") as f:
        for lesson in lessons:
            f.write(json.dumps(lesson, ensure_ascii=False) + "\n")


def _backfill_embeddings(lessons: list, base_url: str, embed_model: str) -> bool:
    """Mutates lessons in place, embedding any entry that doesn't have one
    yet (e.g. taught before an embedding model was available, or before this
    feature existed). Returns True if anything changed."""
    changed = False
    for lesson in lessons:
        if not lesson.get("embedding"):
            vec = _embed(base_url, embed_model, lesson["issue"])
            if vec:
                lesson["embedding"] = vec
                changed = True
    return changed


def _select_lessons(
    prompt: str, category: str, base_url: str, embed_model: str,
    top_k: int, min_similarity: float,
) -> list:
    all_lessons = _load_lessons()
    if embed_model and _backfill_embeddings(all_lessons, base_url, embed_model):
        _save_lessons(all_lessons)

    candidates = all_lessons
    if category:
        candidates = [l for l in candidates if l.get("category") in (category, "general")]
    if not candidates:
        return []

    if not embed_model:
        return candidates[-top_k:]

    query_vec = _embed(base_url, embed_model, prompt)
    if not query_vec:
        return candidates[-top_k:]  # embedding call failed -- degrade to recency

    scored = []
    for lesson in candidates:
        vec = lesson.get("embedding")
        if not vec:
            continue
        sim = _cosine(query_vec, vec)
        if sim >= min_similarity:
            scored.append((sim, lesson))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [lesson for _, lesson in scored[:top_k]]


def _format_lessons_block(lessons: list) -> str:
    if not lessons:
        return ""
    lines = ["Corrections from past mistakes -- do not repeat these:"]
    for lesson in lessons:
        lines.append(f"- [{lesson.get('category', 'general')}] {lesson['issue']} -> {lesson['fix']}")
    return "\n".join(lines)


# ── LM Studio chat completion (stdlib-only HTTP) ───────────────────────────

def _chat(base_url: str, model: str, messages: list, max_tokens: int, timeout: float) -> str:
    payload = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens}).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Could not reach LM Studio at {base_url}: {exc}") from exc
    content = data["choices"][0]["message"]["content"]
    # Reasoning-tuned models are commonly served with a chat template that
    # auto-prepends the <think> opening tag before generation starts, so it
    # never appears in the response -- but the model's own </think> closing
    # tag does. Strip through it so callers get the final answer, not the
    # chain-of-thought.
    if "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    return content


# ── Subcommands ─────────────────────────────────────────────────────────

def cmd_run(args) -> None:
    if args.file:
        # Real-world files (logs especially) are often not clean UTF-8 on
        # Windows -- best-effort decode rather than failing outright.
        with open(args.file, encoding="utf-8", errors="replace") as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read()
    else:
        print("error: provide a prompt argument, --file, or pipe text via stdin.", file=sys.stderr)
        sys.exit(2)

    try:
        _ensure_server(args.base_url)
    except (ConnectionError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Load the (small) embedding model BEFORE the main model, not after.
    # Loading a second model can cause LM Studio to rebalance GPU memory and
    # partially evict a model that's already resident -- observed: a fully
    # GPU-offloaded 21GB LLM got silently pushed partway onto CPU when the
    # embedding model loaded afterward, dropping generation from ~1.5s to
    # 90+s. Loading the tiny embedding model first means the LLM's own
    # --gpu max load (in _ensure_model below) happens last and claims
    # correct offload with the embedding model's footprint already
    # accounted for. This does not protect against an LLM that was already
    # loaded by something else before this run -- if generation seems slow,
    # reload it manually: lms unload --all && lms load <model> --gpu max
    embed_model = ""
    if not args.no_lessons:
        try:
            embed_model = _ensure_embedding_model(args.embed_model, args.ttl)
        except Exception:
            pass  # semantic retrieval is best-effort -- falls back to recency below
        if not embed_model:
            print("note: no embedding model available -- lesson matching falls back "
                  "to most-recent-in-category instead of semantic similarity", file=sys.stderr)

    try:
        model = _ensure_model(args.model, args.ttl)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    system = args.system or ""
    if not args.no_lessons:
        lessons = _select_lessons(
            prompt, args.category, args.base_url, embed_model,
            args.top_lessons, args.min_similarity,
        )
        lessons_text = _format_lessons_block(lessons)
        if lessons_text:
            system = (system + "\n\n" + lessons_text).strip()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        sys.stdout.write(_chat(args.base_url, model, messages, args.max_tokens, args.timeout))
    except ConnectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_teach(args) -> None:
    entry = {"category": args.category, "issue": args.issue, "fix": args.fix}
    try:
        _ensure_server(args.base_url, startup_timeout=15.0)
        embed_model = _ensure_embedding_model(args.embed_model, args.ttl)
        if embed_model:
            vec = _embed(args.base_url, embed_model, args.issue)
            if vec:
                entry["embedding"] = vec
    except Exception:
        pass  # best-effort -- lesson is still recorded; embeds lazily on next `run`

    LESSONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LESSONS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    status = "embedded" if "embedding" in entry else "not embedded yet -- will backfill on next `run`"
    print(f"Recorded lesson under category={args.category!r} ({status}) -> {LESSONS_PATH}")


def cmd_lessons(args) -> None:
    lessons = _load_lessons()
    if args.category:
        lessons = [l for l in lessons if l.get("category") in (args.category, "general")]
    if not lessons:
        print("No lessons recorded yet.")
        return
    for i, lesson in enumerate(lessons, 1):
        embedded = "embedded" if lesson.get("embedding") else "not embedded"
        print(f"{i}. [{lesson.get('category', 'general')}, {embedded}] {lesson['issue']}\n   -> {lesson['fix']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loyal-wingman", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run a prompt against the local model (auto-starts LM Studio if needed)")
    r.add_argument("prompt", nargs="?", help="Prompt text (omit to read from --file or stdin)")
    r.add_argument("--file", help="Read prompt text from this file instead of the positional arg")
    r.add_argument("--system", default=None, help="System prompt")
    r.add_argument("--category", default=None,
                    help="Restrict lesson candidates to this category (plus 'general') before ranking "
                         "by similarity. Omit to search all lessons regardless of category.")
    r.add_argument("--model", default="", help="Model id (default: whatever's loaded, else LOYAL_WINGMAN_MODEL, "
                                                "else the sole downloaded model if unambiguous)")
    r.add_argument("--embed-model", default="", dest="embed_model",
                    help="Embedding model id for lesson retrieval (default: whatever's loaded, else "
                         "LOYAL_WINGMAN_EMBED_MODEL, else the sole downloaded embedding model if unambiguous)")
    r.add_argument("--top-lessons", type=int, default=DEFAULT_TOP_LESSONS, dest="top_lessons",
                    help=f"Max lessons to include, most-similar first (default: {DEFAULT_TOP_LESSONS})")
    r.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY, dest="min_similarity",
                    help=f"Cosine similarity floor for a lesson to be included (default: {DEFAULT_MIN_SIMILARITY})")
    r.add_argument("--no-lessons", action="store_true", dest="no_lessons",
                    help="Skip lesson retrieval entirely (faster; no embedding model needed)")
    r.add_argument("--base-url", default="http://localhost:1234/v1", dest="base_url")
    r.add_argument("--ttl", type=int, default=DEFAULT_TTL_SECONDS,
                    help="Idle seconds before LM Studio auto-unloads the model if it had to be loaded (default: 1800)")
    r.add_argument("--max-tokens", type=int, default=2048, dest="max_tokens",
                    help="Reasoning models spend tokens on chain-of-thought before "
                         "the answer -- keep this generous or the response may be cut "
                         "off mid-thought with no final answer (default: 2048)")
    r.add_argument("--timeout", type=float, default=120.0)
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("teach", help="Record a correction so the model doesn't repeat a mistake")
    t.add_argument("--category", default="general", help="Tag grouping related lessons (default: general)")
    t.add_argument("--issue", required=True, help="What the model got wrong")
    t.add_argument("--fix", required=True, help="What it should have done instead")
    t.add_argument("--embed-model", default="", dest="embed_model", help="Embedding model id (see `run --help`)")
    t.add_argument("--base-url", default="http://localhost:1234/v1", dest="base_url")
    t.add_argument("--ttl", type=int, default=DEFAULT_TTL_SECONDS)
    t.set_defaults(func=cmd_teach)

    l = sub.add_parser("lessons", help="List recorded lessons")
    l.add_argument("--category", default=None, help="Filter by category")
    l.set_defaults(func=cmd_lessons)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
