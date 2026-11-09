# mimOE agent demo for mimik

A small, tool-using AI agent that runs entirely against a local [mimOE
Studio](https://developer.mimik.com/mimOE-studio-early-access-download-v2)
inference endpoint — no cloud API, no network calls, inference happens
on-device.

## The assignment

1. Download and install mimOE Studio from https://developer.mimik.com/mimOE-studio-early-access-download-v2
2. Go to the Model View in the Studio, pick a local model (e.g., SmolLM2 which comes bundled), and load it.
3. There's an API button in the model view — click it and explore the local inference endpoint it exposes. This is an OpenAI-compatible API running entirely on your device.
4. Using the simplest path, the "BYO Framework" approach, build a small AI agent that points to this local mimOE inference endpoint, using whatever framework you're comfortable with (LangChain, LlamaIndex, CrewAI, or just raw API calls). It doesn't need to be complex — what matters is that it works and you can explain your choices.
5. Push the code to a GitHub repo and share the link.

The evaluation is on the ability to explore an unfamiliar AI technology, get it running, connect the pieces, and articulate what was built and why.

**Platform note:** mimOE Studio's graphical Model View/API button is
macOS- and Windows-only. On Linux, mimik's own download page says to use
the CLI installer instead (`developer.mimik.com/docs/ai-foundation`),
which is what's running here — same OpenAI-compatible endpoint, same
model, just reached via terminal/`curl` rather than a GUI panel.

Built as the take-home exercise above. Uses the **"BYO Framework"**
approach: a handful of pure Python functions and
[LiteLLM](https://github.com/BerriAI/litellm) as a thin client, pointed at
mimOE's OpenAI-compatible endpoint. No agent framework (LangChain/CrewAI/
etc.) — the loop is simple enough that a framework would add indirection
without adding value. The agent loop structure (propose → parse → execute
→ update memory → decide) is adapted from a template built during the
Agentic AI Developer specialization (Coursera, Vanderbilt University, Dr.
Jules White).

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
- **No native OpenAI `tools=[...]` calling.** I tested it directly — the
  bundled `smollm2-360m` model does not actually return `tool_calls`; it
  just emits plausible-looking JSON as regular text with
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

## Phrasing sensitivity, and the circuit breaker that followed

`temperature=0` makes the agent deterministic *for a fixed wording* of the
task, but it is not robust *across* wordings. Rephrasing the port-lookup
question (`"what port does mimoE run on? Check workspace files to find
out"` instead of the tested `"What port does mimOE run on? Check the
workspace files to find out."`) was enough to make it read the right file,
get the right content on turn 1 — then keep re-issuing `list_files` /
`read_file` instead of recognizing it already had the answer, burning all
`MAX_ITERATIONS` turns with no final answer. I found this the way these
things actually get found: running the exact demo command with slightly
different wording than the one it was tuned against.

The fix is in two layers, both in `main()`, not in the prompt (a prose rule
telling it not to repeat calls had no measurable effect):

1. **Soft nudge, 2nd identical call:** the code tracks `(tool_name, args)`
   per call. A repeat gets an extra `"note"` field appended to its result —
   the same "feedback the model can act on" pattern already used for
   errors — suggesting it call `terminate` next.
2. **Hard stop, 3rd identical call:** if the nudge doesn't work either, the
   code stops trusting the model's judgment on this one decision and
   force-terminates using the most recent file content it retrieved,
   labeled `(forced: ..., not model-chosen)` in the output so it's never
   mistaken for something the model decided.

This fully fixed the port-lookup case for the problem phrasing above — it
now reliably terminates with the right answer by turn 4 instead of hanging
for 10. It does **not** fully fix the deeper problem: for a task that needs
real synthesis ("how many items are on the shopping list?"), the circuit
breaker still stops the loop, but the model's final message can be
nonsense — in testing it echoed the nudge text itself back as its "answer"
instead of counting the list. The guard rails guarantee the loop
*terminates*; they don't guarantee the answer is *correct*. That second
guarantee is out of reach for a 360M-parameter model without more work — a
larger local model (mimOE supports swapping models in Model View) would be
the first thing to try next.

I'm keeping both the fix and its limits in the repo and in this README
rather than only showing the clean run: the guard rails (iteration cap,
consecutive-error stop, repeat circuit breaker) doing their job — stopping
a run that can't make progress instead of hanging or crashing — is the
actual engineering lesson here, independent of whether every task this
tiny model is given gets a correct answer.

## Demo: a working run

Verified on the installed mimOE instance (`v3.20.2`, developer edition,
installed via the Linux CLI quick-start):

```
$ ./status.sh
mimoe [pid = 165216, version = v3.20.2 (developer edition)] is running

$ curl -X POST "http://localhost:8083/mimik-ai/openai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 1234" \
  -d '{"model": "smollm2-360m", "messages": [{"role": "user", "content": "Complete this sentence: AI is like a"}]}'
```

`workspace/about_mimoe.txt` contains a fact the model cannot already know
(the local port mimOE happened to start on) — the demo is designed so a
right answer is proof the agent actually read the file, not a lucky guess
from pretraining.

```
$ source .venv/bin/activate
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
grounded answer. A slightly reworded version of the same question (see
"Phrasing sensitivity" above) still gets the right answer, just via the
circuit breaker at turn 4 instead of the model terminating on its own:

```
What should the agent do? what port does mimoE run on? Check workspace files to find out
[0] list_files -> {'files': ['about_mimoe.txt', 'shopping_list.txt']}  (seeded, not model-chosen)
[1] read_file -> {'content': 'mimOE local port: 8083\n'}
[2] list_files -> {..., 'note': 'You already made this exact call earlier...'}
[3] read_file -> {..., 'note': 'You already made this exact call earlier...'}
[4] terminate -> {'message': 'mimOE local port: 8083\n'}  (forced: repeated identical call 3x, not model-chosen)
```

## Running it

```bash
# 1. Start mimOE Studio, load smollm2-360m in Model View, click the API
#    button to confirm the base URL/model name (defaults below assume the
#    defaults mimOE itself prints). On Linux, use the CLI quick-start
#    instead (developer.mimik.com/docs/ai-foundation) — same endpoint.

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
| `read_file` resolves and checks `os.path.commonpath` against `workspace/` | Caught during testing: `read_file("../agent.py")` originally escaped the workspace sandbox and read arbitrary files on disk |
| `generate_response` failures (`openai.APIError`) caught in `main()` with a clear message and `sys.exit(1)` | Caught during testing: if mimOE isn't running, the uncaught exception was a raw LiteLLM/OpenAI stack trace instead of an actionable "is mimOE running?" message |
| Repeated identical tool calls get a steering note (2nd time), then a forced, code-driven `terminate` (3rd time) | Caught during testing: a rephrased task made the model loop to `MAX_ITERATIONS` instead of recognizing it already had the answer; see "Phrasing sensitivity" above |
