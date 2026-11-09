"""
Unit tests for the agent's pure plumbing: parsing model replies into actions
and dispatching actions to tools. These do not call the mimOE endpoint, so
they run without mimOE installed or running.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import extract_markdown_block, list_files, parse_action, read_file, run_action


class TestExtractMarkdownBlock(unittest.TestCase):
    def test_extracts_fenced_block(self):
        text = 'blah\n```action\n{"a": 1}\n```\nblah'
        self.assertEqual(extract_markdown_block(text, "action"), '{"a": 1}')

    def test_returns_none_when_fence_missing(self):
        self.assertIsNone(extract_markdown_block("no fence here", "action"))


class TestParseAction(unittest.TestCase):
    def test_valid_action(self):
        reply = '```action\n{"tool_name": "list_files", "args": {}}\n```'
        self.assertEqual(parse_action(reply), {"tool_name": "list_files", "args": {}})

    def test_missing_fence_becomes_error_action(self):
        action = parse_action("just talking, no action block")
        self.assertEqual(action["tool_name"], "error")
        self.assertIn("message", action["args"])

    def test_invalid_json_becomes_error_action(self):
        action = parse_action("```action\nnot json\n```")
        self.assertEqual(action["tool_name"], "error")

    def test_missing_required_key_becomes_error_action(self):
        action = parse_action('```action\n{"tool_name": "list_files"}\n```')
        self.assertEqual(action["tool_name"], "error")


class TestRunAction(unittest.TestCase):
    def test_list_files_returns_workspace_contents(self):
        result = run_action({"tool_name": "list_files", "args": {}})
        self.assertEqual(set(result["files"]), set(list_files()))

    def test_read_file_returns_content(self):
        result = run_action({"tool_name": "read_file", "args": {"file_name": "about_mimoe.txt"}})
        self.assertIn("8083", result["content"])

    def test_read_file_missing_file_returns_error(self):
        result = run_action({"tool_name": "read_file", "args": {"file_name": "does_not_exist.txt"}})
        self.assertIn("error", result)
        self.assertIn("about_mimoe.txt", result["error"])  # error lists available files

    def test_read_file_missing_arg_returns_error(self):
        result = run_action({"tool_name": "read_file", "args": {}})
        self.assertIn("error", result)

    def test_terminate_returns_message(self):
        result = run_action({"tool_name": "terminate", "args": {"message": "done"}})
        self.assertEqual(result["message"], "done")

    def test_unknown_tool_returns_error(self):
        result = run_action({"tool_name": "fly_to_the_moon", "args": {}})
        self.assertIn("error", result)


class TestReadFileDirect(unittest.TestCase):
    def test_raises_file_not_found_for_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            read_file("nope.txt")

    def test_blocks_path_traversal_outside_workspace(self):
        with self.assertRaises(FileNotFoundError):
            read_file("../agent.py")


if __name__ == "__main__":
    unittest.main()
