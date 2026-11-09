"""
A small tool-using agent that runs entirely against a local mimOE Studio
inference endpoint (OpenAI-compatible, on-device, no cloud calls).

Agent loop (see README.md for the full explanation):
  1. construct prompt  = agent rules + memory
  2. generate response = ask the model
  3. parse response     = extract a ```action block into {tool_name, args}
  4. execute action      = run the matching Python function
  5. update memory       = assistant gets the raw reply, user gets the result
  6. decide whether to continue (terminate / error streak / iteration cap)
"""

import json
import os
import sys
from typing import Dict, List, Optional

from litellm import completion
from openai import APIError

# --- mimOE connection settings -------------------------------------------
# mimOE Studio exposes an OpenAI-compatible endpoint on-device. The model
# name and base URL come straight from the API panel in Model View.
API_BASE = os.environ.get("MIMOE_API_BASE", "http://localhost:8083/mimik-ai/openai/v1")
API_KEY = os.environ.get("MIMOE_API_KEY", "1234")  # local placeholder token mimOE itself prints; not a real secret
MODEL = f"openai/{os.environ.get('MIMOE_MODEL', 'smollm2-360m')}"

MAX_ITERATIONS = 10
MAX_CONSECUTIVE_ERRORS = 2
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")

# SmolLM2-360M is too small to reliably follow instructions from prose rules
# alone (confirmed experimentally: it rambled past the ```action format
# entirely). It imitates a worked example much more reliably, so the rules
# are followed by one complete, fixed example conversation showing the full
# list_files -> read_file -> terminate cycle. This primer is prepended to
# every prompt (see main()), ahead of the model's real, growing memory.
AGENT_RULES = [
    {
        "role": "system",
        "content": (
            "You are a small autonomous agent with access to these tools:\n"
            "- list_files(): returns the names of files in the workspace.\n"
            "- read_file(file_name): returns the text content of one file.\n"
            "- terminate(message): ends the task and reports your final answer.\n\n"
            "Always reply with exactly one ```action code block containing JSON, "
            "and nothing else. Below is one worked example, then your real task."
        ),
    },
    {"role": "user", "content": "EXAMPLE TASK: What is in config.txt?"},
    {"role": "assistant", "content": '```action\n{"tool_name": "list_files", "args": {}}\n```'},
    {"role": "user", "content": json.dumps({"files": ["config.txt"]})},
    {
        "role": "assistant",
        "content": '```action\n{"tool_name": "read_file", "args": {"file_name": "config.txt"}}\n```',
    },
    {"role": "user", "content": json.dumps({"content": "timeout=30"})},
    {
        "role": "assistant",
        "content": '```action\n{"tool_name": "terminate", "args": {"message": "config.txt sets timeout=30."}}\n```',
    },
]


def generate_response(messages: List[Dict]) -> str:
    """Send the conversation to the local mimOE model and return the reply text."""
    response = completion(
        model=MODEL,
        api_base=API_BASE,
        api_key=API_KEY,
        messages=messages,
        max_tokens=150,
        temperature=0,  # SmolLM2-360M strays from the action format at higher temperatures
    )
    finish = response.choices[0].finish_reason
    print(f"[finish_reason: {finish}]")
    if finish != "stop":
        print("WARNING: the reply may be incomplete or refused.")
    return response.choices[0].message.content or ""


def extract_markdown_block(text: str, tag: str) -> Optional[str]:
    """Return the content of a ```<tag> ... ``` fenced block, or None if absent."""
    fence = f"```{tag}"
    if fence not in text:
        return None
    return text.split(fence, 1)[1].split("```", 1)[0].strip()


def parse_action(response: str) -> Dict:
    """Extract {tool_name, args} from a reply. Any failure becomes an 'error'
    action carrying a message written for the model, so the next turn can
    self-correct instead of crashing the loop."""
    block = extract_markdown_block(response, "action")
    if block is None:
        return {
            "tool_name": "error",
            "args": {
                "message": (
                    "Your reply had no ```action block. Reply with exactly one "
                    'block in this form: ```action\n{"tool_name": "...", "args": {...}}\n```'
                )
            },
        }
    try:
        action = json.loads(block)
    except json.JSONDecodeError as e:
        return {"tool_name": "error", "args": {"message": f"Your ```action block was not valid JSON: {e}"}}
    if not isinstance(action, dict) or "tool_name" not in action or "args" not in action:
        return {
            "tool_name": "error",
            "args": {"message": 'Your action JSON must be an object with "tool_name" and "args" keys.'},
        }
    return action


def list_files() -> List[str]:
    return sorted(os.listdir(WORKSPACE_DIR))


def read_file(file_name: str) -> str:
    workspace_root = os.path.realpath(WORKSPACE_DIR)
    path = os.path.realpath(os.path.join(workspace_root, file_name))
    not_found = FileNotFoundError(
        f"'{file_name}' does not exist. Available files: {', '.join(list_files()) or '(none)'}"
    )
    # Reject paths that escape the workspace (e.g. "../agent.py") rather than
    # letting a path-traversal argument read anything on disk.
    if os.path.commonpath([workspace_root, path]) != workspace_root:
        raise not_found
    if not os.path.isfile(path):
        raise not_found
    with open(path) as f:
        return f.read()


def run_action(action: Dict) -> Dict:
    """The tool switchboard. The model decides what to call; this function
    decides how to execute it, and turns any failure into a result the
    model can read and react to on the next turn."""
    tool_name = action.get("tool_name")
    args = action.get("args", {})
    try:
        if tool_name == "list_files":
            return {"files": list_files()}
        elif tool_name == "read_file":
            return {"content": read_file(args["file_name"])}
        elif tool_name == "terminate":
            return {"message": args.get("message", "Task complete.")}
        elif tool_name == "error":
            return {"error": args.get("message", "Unknown error.")}
        else:
            return {"error": f"Unknown tool_name '{tool_name}'. Available tools: list_files, read_file, terminate."}
    except KeyError as e:
        return {"error": f"Missing required argument {e} for tool '{tool_name}'."}
    except FileNotFoundError as e:
        return {"error": str(e)}


def main() -> None:
    task = input("What should the agent do? ").strip()

    # SmolLM2-360M is reliable at *choosing the next tool given file names it
    # already has*, but experimentally it often skips straight to a
    # hallucinated terminate() on a fresh task instead of looking anything
    # up first. Rather than trust a 360M-parameter model's judgment for a
    # deterministic bootstrap step, the code calls list_files() itself and
    # seeds it into memory as if it were the agent's own first move. Every
    # decision after that point (which file to read, when it has enough
    # information, when to terminate) is genuinely made by the model.
    bootstrap_files = list_files()
    memory = [
        {"role": "user", "content": task},
        {"role": "assistant", "content": '```action\n{"tool_name": "list_files", "args": {}}\n```'},
        {"role": "user", "content": json.dumps({"files": bootstrap_files})},
    ]
    print(f"[0] list_files -> {{'files': {bootstrap_files}}}  (seeded, not model-chosen)")
    consecutive_errors = 0

    for i in range(MAX_ITERATIONS):
        prompt = AGENT_RULES + memory                    # 1. construct prompt
        try:
            response = generate_response(prompt)          # 2. generate
        except APIError as e:
            print(f"Could not reach mimOE at {API_BASE} ({e}).")
            print("Is mimOE running? Check with ./status.sh, start it with ./start.sh.")
            sys.exit(1)
        action = parse_action(response)                   # 3. parse
        result = run_action(action)                        # 4. execute
        print(f"[{i + 1}] {action.get('tool_name')} -> {result}")

        memory.extend([                                   # 5. update memory
            {"role": "assistant", "content": response},
            {"role": "user", "content": json.dumps(result)},
        ])

        consecutive_errors = consecutive_errors + 1 if "error" in result else 0
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            print(f"Stopped: {MAX_CONSECUTIVE_ERRORS} errors in a row.")
            break

        if action.get("tool_name") == "terminate":        # 6. continue?
            break
    else:
        print("Stopped: reached the iteration limit.")


if __name__ == "__main__":
    main()
