# mimOE agent demo for MIMIK 

A small, tool-using AI agent that runs entirely against a local [mimOE
Studio](https://developer.mimik.com/mimOE-studio-early-access-download-v2)
inference endpoint — no cloud API, no network calls, inference happens
on-device.
Instructions: 
1. Download and install mimOE Studio from https://developer.mimik.com/mimOE-studio-early-access-download-v2

2. Go to the Model View in the Studio, pick a local model (e.g., SmolLM2 which comes bundled), and load it.

3. You'll notice there's an API button in the model view. Click it and explore the local inference endpoint it exposes. This is an OpenAI-compatible API running entirely on your device.

4. Using the simplest path, the "BYO Framework" approach, build a small AI agent that points to this local mimOE inference endpoint. You can use whatever framework you're comfortable with (LangChain, LlamaIndex, CrewAI, or even just raw API calls). The agent doesn't need to be complex. What matters is that it works and you can explain your choices.

5. Push your code to a GitHub repo and share the link with us.

The key thing we're evaluating is your ability to explore an unfamiliar AI technology, get it running, connect the pieces, and articulate what you built and why. We fully expect you to use AI coding assistants (Claude, Copilot, etc.) to help. That's part of the skillset we value.

The task is to build a take-home exercise that uses the **"BYO Framework"** approach: a
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

This is coming from a template that I acquired from an Agentic AI course with Vanderbilt University. 
This is the terminal information of the system tested and runned inside my laptop: 

============================================
  mimOE AI Foundation is ready!
============================================

Test your setup with this command:

curl -X POST "http://localhost:8083/mimik-ai/openai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 1234" \
  -d '{
    "model": "smollm2-360m",
    "messages": [{"role": "user", "content": "Complete this sentence: AI is like a"}]
  }'

To stop mimOE (universal):    pkill -f mimoe

 cd to the mimOE installed directory 
  cd /home/oscar/mimo_interview

To stop mimOE (local):        ./stop.sh
To restart (background):      ./stop.sh >/dev/null; ./start.sh >/dev/null 2>&1 & sleep 2; ./status.sh
To restart (interactive):      ./stop.sh ; ./start.sh
To get mimoe status:          ./status.sh
To view logs (follow):        tail -f ./.edge/logs/mimoe.log

Documentation: https://developer.mimik.com/docs/ai-foundation

oscar@oscar-Yoga-7-15ITL5:~/mimo_interview$ source .venv/bin/activate
(.venv) oscar@oscar-Yoga-7-15ITL5:~/mimo_interview$ python3 agent.py
What should the agent do? what port does mimoE run on? Check workspace files to find out 
[0] list_files -> {'files': ['about_mimoe.txt', 'shopping_list.txt']}  (seeded, not model-chosen)
[finish_reason: stop]
[1] read_file -> {'content': 'mimOE local port: 8083\n'}
[finish_reason: stop]
[2] list_files -> {'files': ['about_mimoe.txt', 'shopping_list.txt']}
[finish_reason: stop]
[3] read_file -> {'content': 'mimOE local port: 8083\n'}
[finish_reason: stop]
[4] list_files -> {'files': ['about_mimoe.txt', 'shopping_list.txt']}
[finish_reason: stop]
[5] read_file -> {'content': 'mimOE local port: 8083\n'}
[finish_reason: stop]
[6] list_files -> {'files': ['about_mimoe.txt', 'shopping_list.txt']}
[finish_reason: stop]
[7] read_file -> {'content': 'mimOE local port: 8083\n'}
[finish_reason: stop]
[8] list_files -> {'files': ['about_mimoe.txt', 'shopping_list.txt']}
[finish_reason: stop]
[9] read_file -> {'content': 'mimOE local port: 8083\n'}
[finish_reason: stop]
[10] list_files -> {'files': ['about_mimoe.txt', 'shopping_list.txt']}
Stopped: reached the iteration limit.
(.venv) oscar@oscar-Yoga-7-15ITL5:~/mimo_interview$ python3 -m unittest tests.test_agent -v
test_extracts_fenced_block (tests.test_agent.TestExtractMarkdownBlock.test_extracts_fenced_block) ... ok
test_returns_none_when_fence_missing (tests.test_agent.TestExtractMarkdownBlock.test_returns_none_when_fence_missing) ... ok
test_invalid_json_becomes_error_action (tests.test_agent.TestParseAction.test_invalid_json_becomes_error_action) ... ok
test_missing_fence_becomes_error_action (tests.test_agent.TestParseAction.test_missing_fence_becomes_error_action) ... ok
test_missing_required_key_becomes_error_action (tests.test_agent.TestParseAction.test_missing_required_key_becomes_error_action) ... ok
test_valid_action (tests.test_agent.TestParseAction.test_valid_action) ... ok
test_blocks_path_traversal_outside_workspace (tests.test_agent.TestReadFileDirect.test_blocks_path_traversal_outside_workspace) ... ok
test_raises_file_not_found_for_missing_file (tests.test_agent.TestReadFileDirect.test_raises_file_not_found_for_missing_file) ... ok
test_list_files_returns_workspace_contents (tests.test_agent.TestRunAction.test_list_files_returns_workspace_contents) ... ok
test_read_file_missing_arg_returns_error (tests.test_agent.TestRunAction.test_read_file_missing_arg_returns_error) ... ok
test_read_file_missing_file_returns_error (tests.test_agent.TestRunAction.test_read_file_missing_file_returns_error) ... ok
test_read_file_returns_content (tests.test_agent.TestRunAction.test_read_file_returns_content) ... ok
test_terminate_returns_message (tests.test_agent.TestRunAction.test_terminate_returns_message) ... ok
test_unknown_tool_returns_error (tests.test_agent.TestRunAction.test_unknown_tool_returns_error) ... ok

----------------------------------------------------------------------
Ran 14 tests in 0.001s

OK
(.venv) oscar@oscar-Yoga-7-15ITL5:~/mimo_interview$ deactivate
