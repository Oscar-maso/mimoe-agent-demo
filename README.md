# mimOE agent demo

A small, tool-using AI agent that runs entirely against a local [mimOE
Studio](https://developer.mimik.com/mimOE-studio-early-access-download-v2)
inference endpoint — no cloud API, no network calls, inference happens
on-device.

Built as a take-home exercise. Uses the **"BYO Framework"** approach: a
handful of pure Python functions and [LiteLLM](https://github.com/BerriAI/litellm)
as a thin client, pointed at mimOE's OpenAI-compatible endpoint. No agent
framework (LangChain/CrewAI/etc.) — the loop is simple enough that a
framework would add indirection without adding value.

## What it does

`agent.py` implements the classic agent loop: the model proposes one action
per turn as JSON, the code executes it and feeds the result back, and this
repeats until the model terminates, two tool calls error in a row, or an
iteration cap is hit.

Three tools, scoped to a `workspace/` folder:

- `list_files()` — names of files in the workspace
- `read_file(file_name)` — a file's contents
- `terminate(message)` — ends the task with a final answer

```
construct prompt → generate response → parse into action → execute action
      ↑                                                          |
      └──────────────── update memory, decide to continue ───────┘
```

The model never executes anything itself — it only ever proposes
`{"tool_name": ..., "args": ...}` as text. The code decides whether that's
a valid call, runs the real Python function, and reports the result back as
a new `user` message (it's new information from outside the model, not
something the model said — tagging it `assistant` would break turn
alternation and make the model think it authored the data itself).

## Why mimOE + LiteLLM + this loop shape

- **mimOE's endpoint is OpenAI-compatible**, so any OpenAI-style client
  works unmodified except for `api_base`/`api_key`. LiteLLM was the natural
  choice since it's what I'd already used for Claude via its `anthropic/...`
  model prefix in coursework — swapping to a local server is a one-line
  change (`model="openai/smollm2-360m"`, `api_base=...`).
- **No native OpenAI `tools=[...]` calling.** I tested it directly (see
  below) — the bundled `smollm2-360m` model does not actually return
  `tool_calls`; it just emits plausible-looking JSON as regular text with
  `finish_reason: "stop"`. So the agent uses the course's prompt-based
  `\`\`\`action` block convention instead, which only requires the model to
  produce parseable text, not a provider-specific tool-calling feature.
- **The propose/execute split** (model proposes, code executes) is a
  safety and testability boundary, not a convenience: the code can validate
  arguments, catch missing files, and reject nonsense before anything runs,
  regardless of what the model outputs.

## A real limitation I hit, and how I worked around it

`smollm2-360m` is a 360-million-parameter model running on a 2048-token
context. It is **not** reliable at following instructions from prose rules
alone — confirmed by testing: with only a system prompt describing the
format, it ignored it entirely and rambled about unrelated things until it
hit `finish_reason: "length"`.

Two adjustments fixed most of this:

1. **One fixed few-shot example** (a full `list_files → read_file →
   terminate` cycle) prepended to every prompt. Small models imitate a
   worked example far more reliably than they follow abstract instructions
   — this got JSON-format compliance to effectively 100% in testing.
2. **`temperature=0`.** At default sampling it would occasionally pick a
   plausible-sounding but wrong tool name or invent argument keys that
   don't exist. Determinism matters more than creativity for a planner this
   small.

One thing the few-shot example and `temperature=0` did *not* fix: on a
fresh task, the model would often skip straight to `terminate` with a
confidently wrong, hallucinated answer instead of calling `list_files`
first. Rather than trust a 360M-parameter model's judgment for a
deterministic bootstrap step, `main()` calls `list_files()` itself and
seeds it into memory as the agent's first "turn" before the model is asked
to decide anything. Every decision after that point — which file to read,
when to stop — is genuinely made by the model, not scripted.

### What still doesn't work reliably

Asking it to *synthesize* an answer (e.g. "how many items are on the
shopping list?") makes it loop between `list_files` and `read_file` without
ever calling `terminate`, until the iteration cap stops it:

```
$ echo "How many items are on the shopping list?" | python3 agent.py
[0] list_files -> {'files': ['about_mimoe.txt', 'shopping_list.txt']}  (seeded, not model-chosen)
[1] read_file -> {'content': 'Shopping list\n...'}
[2] list_files -> {'files': [...]}
[3] read_file -> {'content': 'Shopping list\n...'}
...
Stopped: reached the iteration limit.
```

I'm keeping this in the repo rather than cherry-picking only the success
case: it's a genuine finding about the limits of a 360M on-device model,
and the guard rail (`MAX_ITERATIONS`) did exactly its job — it stopped the
run cleanly instead of hanging or crashing. A direct, single-fact lookup
(below) works reliably and deterministically; anything requiring the model
to reason over or summarize retrieved content does not, at this model
size. A larger local model (mimOE supports swapping models in Model View)
would be the first thing to try next.

## Demo: a working run

`workspace/about_mimoe.txt` contains a fact the model cannot already know
(the local port mimOE happened to start on) — the demo is designed so a
right answer is proof the agent actually read the file, not a lucky guess
from pretraining.

```
$ python3 agent.py
What should the agent do? What port does mimOE run on? Check the workspace files to find out.
[0] list_files -> {'files': ['about_mimoe.txt', 'shopping_list.txt']}  (seeded, not model-chosen)
[finish_reason: stop]
[1] read_file -> {'content': 'mimOE local port: 8083\n'}
[finish_reason: stop]
[2] terminate -> {'message': 'mimOE local port: 8083.\n'}
```

Three turns: the model chose `about_mimoe.txt` over the decoy
`shopping_list.txt` on its own, read it, and terminated with the correct,
grounded answer.

## Running it

```bash
# 1. Start mimOE Studio, load smollm2-360m in Model View, click the API
#    button to confirm the base URL/model name (defaults below assume the
#    defaults mimOE itself prints).

# 2. From this directory:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the agent:
python3 agent.py
# What should the agent do? What port does mimOE run on? Check the workspace files to find out.

# 4. Run the tests (no mimOE/model required — pure plumbing only):
python3 -m unittest tests.test_agent -v
```

Connection settings are overridable via environment variables if your
mimOE instance differs:

| Variable         | Default                                       |
|------------------|------------------------------------------------|
| `MIMOE_API_BASE` | `http://localhost:8083/mimik-ai/openai/v1`     |
| `MIMOE_API_KEY`  | `1234` (mimOE's own local placeholder token — not a real credential) |
| `MIMOE_MODEL`    | `smollm2-360m`                                 |

## Design decisions, summarized

| Decision | Why |
|---|---|
| LiteLLM over raw `openai` SDK | I'd already used it for Claude; same call shape, one-line swap of `model`/`api_base` to point at mimOE instead |
| Prompt-based `\`\`\`action` blocks, not `tools=[...]` | Verified mimOE's bundled model doesn't actually return `tool_calls` |
| One fixed few-shot example in the system context | Prose rules alone were ignored; a worked example wasn't |
| `temperature=0` | Reliability over creativity for a planner this small |
| Scripted first `list_files()` call | The model reliably chooses well *given* file names, but not reliably reaches for a tool at all on a fresh task |
| Errors returned as `{"error": ...}` results, not exceptions that crash the loop | The model can read an error and react to it next turn; two in a row stops the run instead of looping forever |
| `read_file` raises, `run_action` catches | Keeps the "did this fail" decision in one place rather than overloading the return value's shape |
