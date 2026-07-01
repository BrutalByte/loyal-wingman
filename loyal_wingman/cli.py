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
MAX_LESSONS_IN_PROMPT = 20
DEFAULT_TTL_SECONDS = 1800
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


def _available_llms() -> list:
    """Locally downloaded LLMs (excludes embedding models)."""
    try:
        result = subprocess.run(
            [_lms_bin(), "ls", "--json"],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(result.stdout or "[]")
        return [m["modelKey"] for m in data if m.get("type") == "llm"]
    except Exception:
        return []


def _ensure_model(model: str, ttl: int, load_timeout: float = 180.0) -> str:
    loaded_ids = [m.get("identifier") for m in _loaded_models()]
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


# ── Lessons store ─────────────────────────────────────────────────────────

def _load_lessons(category: str = None) -> list:
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
    if category:
        lessons = [l for l in lessons if l.get("category") in (category, "general")]
    return lessons


def _lessons_block(category: str = None) -> str:
    lessons = _load_lessons(category)[-MAX_LESSONS_IN_PROMPT:]
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
        model = _ensure_model(args.model, args.ttl)
    except (ConnectionError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    system = args.system or ""
    lessons = _lessons_block(args.category)
    if lessons:
        system = (system + "\n\n" + lessons).strip()

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
    LESSONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {"category": args.category, "issue": args.issue, "fix": args.fix}
    with open(LESSONS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Recorded lesson under category={args.category!r} ({LESSONS_PATH})")


def cmd_lessons(args) -> None:
    lessons = _load_lessons(args.category)
    if not lessons:
        print("No lessons recorded yet.")
        return
    for i, lesson in enumerate(lessons, 1):
        print(f"{i}. [{lesson.get('category', 'general')}] {lesson['issue']}\n   -> {lesson['fix']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loyal-wingman", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run a prompt against the local model (auto-starts LM Studio if needed)")
    r.add_argument("prompt", nargs="?", help="Prompt text (omit to read from --file or stdin)")
    r.add_argument("--file", help="Read prompt text from this file instead of the positional arg")
    r.add_argument("--system", default=None, help="System prompt")
    r.add_argument("--category", default=None, help="Lesson category to apply (default: all recorded lessons)")
    r.add_argument("--model", default="", help="Model id (default: whatever's loaded, else LOYAL_WINGMAN_MODEL, "
                                                "else the sole downloaded model if unambiguous)")
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
