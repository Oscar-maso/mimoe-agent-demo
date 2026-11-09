# mimOE agent demo for mimik

A small, tool-using agent that runs entirely against a local [mimOE
Studio](https://developer.mimik.com/mimOE-studio-early-access-download-v2)
inference endpoint. No cloud API, no network calls, inference happens
on-device.

## The assignment

1. Download and install mimOE Studio from https://developer.mimik.com/mimOE-studio-early-access-download-v2
2. Go to the Model View in the Studio, pick a local model (e.g., SmolLM2 which comes bundled), and load it.
3. There's an API button in the model view. Click it and explore the local inference endpoint it exposes (an OpenAI-compatible API running entirely on your device).
4. Build a small AI agent pointed at that endpoint, using the simplest "BYO Framework" approach, with whatever framework you're comfortable with (LangChain, LlamaIndex, CrewAI, or just raw API calls). Doesn't need to be complex, just needs to work and be explainable.
5. Push the code to a GitHub repo and share the link.

**Platform note:** mimOE Studio's graphical Model View/API button is
macOS and Windows only. On Linux, mimik's own download page points you
to the CLI installer instead (`developer.mimik.com/docs/ai-foundation`).
That's what's actually running here, same endpoint and model, just reached
through a terminal instead of a GUI panel.

## What it does and why

Uses the **BYO Framework** path: a handful of plain Python functions plus
[LiteLLM](https://github.com/BerriAI/litellm) as a thin client pointed at
mimOE. No agent framework, the loop is simple enough that LangChain/CrewAI
would just add indirection. I picked LiteLLM mainly because I already use
it for Claude through its `anthropic/...` model prefix, so pointing it at
a local server instead is a one-line change (`model="openai/smollm2-360m"`,
`api_base=...`). The agent loop itself (propose → parse → execute →
update memory → decide) comes from a template I built in the Agentic AI
Developer specialization on Coursera (Vanderbilt, Dr. Jules White).

`agent.py` runs that loop: the model proposes one action per turn as JSON,
the code executes it and feeds the result back, and this repeats until the
model terminates, two tool calls error in a row, or the iteration cap is
hit. Three tools, scoped to a `workspace/` folder:

- `list_files()` — names of files in the workspace
- `read_file(file_name)` — a file's contents
- `terminate(message)` — ends the task with a final answer

```
construct prompt → generate response → parse into action → execute action
      ↑                                                          |
      └──────────────── update memory, decide to continue ───────┘
```

The model never actually executes anything, it only proposes
`{"tool_name": ..., "args": ...}` as text. The code decides if that's a
valid call, runs the real function, and reports the result back as a new
`user` message. Tool results go in as `user`, not `assistant`, because
they're external information, not something the model said (and because
tagging them `assistant` would break turn alternation).

I didn't bother with OpenAI's native `tools=[...]` calling. Tested it
directly and the bundled `smollm2-360m` model doesn't actually return
`tool_calls`, it just emits JSON-shaped text with `finish_reason: "stop"`.
So the agent uses the course's prompt-based `\`\`\`action` block instead,
which only needs the model to produce parseable text.

## Getting a 360M model to behave

`smollm2-360m` is tiny (360M params, 2048-token context) and it shows.
With just a system prompt describing the format, it ignored it completely
and rambled until it hit `finish_reason: "length"`.

What actually worked:

- **One fixed few-shot example** (`list_files → read_file → terminate`)
  prepended to every prompt. It imitates a worked example far better than
  it follows instructions, this alone got format compliance close to 100%.
- **`temperature=0`.** At default sampling it would pick a plausible but
  wrong tool, or invent argument keys that don't exist.
- **Scripting the first `list_files()` call.** Even with the few-shot
  example, on a fresh task it would often skip straight to `terminate`
  with a confidently wrong, hallucinated answer instead of looking
  anything up. So `main()` calls `list_files()` itself and seeds it into
  memory before the model is asked to decide anything. Everything after
  that (which file to read, when to stop) is genuinely the model's call.

None of this is fully reliable, though. `temperature=0` is deterministic
for a *fixed* wording of a task, but not across wordings, I found this
when a teammate ran the exact same demo with slightly different phrasing
and it read the right file on turn 1, then just kept re-listing and
re-reading instead of terminating, burning all 10 iterations.

Fix ended up being two more layers in `main()` (prose rules telling it
not to repeat itself didn't do anything):

1. A repeated identical tool call gets a `"note"` in its result nudging it
   toward `terminate`.
2. If that doesn't land either, the third repeat force-terminates using
   the last content retrieved, and is logged as `(forced: ..., not
   model-chosen)` so it's never mistaken for the model's own decision.

That fixed the rephrased port-lookup case (now converges by turn 4). It
didn't fix the deeper issue: ask it something that needs actual synthesis,
like counting items in a list, and the circuit breaker still cuts the loop
short, but the final answer can be nonsense (in one run it literally
echoed the nudge text back as its "answer"). The guard rails guarantee the
loop stops. They don't guarantee the answer's right. A bigger local model
would be the obvious next thing to try.

## Demo

Verified against the installed mimOE instance (`v3.20.2`, dev edition,
Linux CLI install):

```
$ ./status.sh
mimoe [pid = 165216, version = v3.20.2 (developer edition)] is running

$ curl -X POST "http://localhost:8083/mimik-ai/openai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 1234" \
  -d '{"model": "smollm2-360m", "messages": [{"role": "user", "content": "Complete this sentence: AI is like a"}]}'
```

`workspace/about_mimoe.txt` holds a fact the model can't already know (the
local port mimOE happened to start on), so a right answer actually proves
the agent read the file instead of guessing from pretraining:

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

It picked `about_mimoe.txt` over the decoy `shopping_list.txt` on its own,
read it, and terminated with the right, grounded answer. The reworded
version of the same question still lands on 8083, just through the
circuit breaker at turn 4 instead of the model stopping itself:

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
#    button to confirm the base URL/model name. On Linux, use the CLI
#    quick-start instead (developer.mimik.com/docs/ai-foundation).

# 2. From this directory:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the agent:
python3 agent.py
# What should the agent do? What port does mimOE run on? Check the workspace files to find out.

# 4. Run the tests (pure plumbing, no mimOE/model needed):
python3 -m unittest tests.test_agent -v
```

Overridable via env vars if your mimOE setup differs:

| Variable         | Default                                       |
|------------------|------------------------------------------------|
| `MIMOE_API_BASE` | `http://localhost:8083/mimik-ai/openai/v1`     |
| `MIMOE_API_KEY`  | `1234` (mimOE's own local placeholder, not a real credential) |
| `MIMOE_MODEL`    | `smollm2-360m`                                 |

## A couple of bugs testing turned up

- `read_file("../agent.py")` originally escaped the `workspace/` sandbox
  and read anything on disk. Fixed by resolving the real path and
  checking it with `os.path.commonpath`.
- If mimOE wasn't running, `generate_response` blew up with a raw
  LiteLLM/OpenAI stack trace. Now caught in `main()` and turned into "is
  mimOE running? try ./status.sh" plus a clean exit.

Both have regression tests in `tests/test_agent.py`.
