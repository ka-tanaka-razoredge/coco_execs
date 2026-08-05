import unittest
from unittest.mock import Mock, patch

import agent_api
from tools.wiki_it import BasicAuthTransport


class RetrieveKnowledgeTests(unittest.TestCase):
    @patch("agent_api.load_local_knowledge")
    @patch("agent_api.requests.get")
    @patch.dict("agent_api.os.environ", {"MONGODB_ACCESSOR_AUTHORIZATION": "test"})
    def test_merges_local_yaml_with_mongodb_knowledge_and_preferences(
        self, get: Mock, load_local_knowledge: Mock
    ) -> None:
        knowledge_response = Mock()
        knowledge_response.json.return_value = [
            {"title": "Remote knowledge", "content": "unrelated"}
        ]
        preferences_response = Mock()
        preferences_response.json.return_value = [
            {"title": "Preference", "content": "example"}
        ]
        get.side_effect = [knowledge_response, preferences_response]
        load_local_knowledge.return_value = [
            {"_source": "knowledge/network.yaml", "content": {"host": "example"}}
        ]

        knowledge = agent_api.retrieve_knowledge("example")

        self.assertEqual(
            knowledge,
            [
                {"title": "Preference", "content": "example"},
                {
                    "_source": "knowledge/network.yaml",
                    "content": {"host": "example"},
                }
            ],
        )
        self.assertEqual(
            [call.args[0] for call in get.call_args_list],
            [
                "https://razor-edge.net/apis/mongodb-accessor/knowledge",
                "https://razor-edge.net/apis/mongodb-accessor/preferences",
            ],
        )
        knowledge_response.raise_for_status.assert_called_once_with()
        preferences_response.raise_for_status.assert_called_once_with()

    def test_loads_repository_yaml_with_its_source_path(self) -> None:
        documents = agent_api.load_local_knowledge()

        self.assertTrue(
            any(document["_source"] == "knowledge/00000100.yaml" for document in documents)
        )


class ToolDispatchTests(unittest.TestCase):
    def test_wiki_it_sends_basic_authentication(self) -> None:
        connection = Mock()

        BasicAuthTransport("user", "password").send_headers(connection, [])

        connection.putheader.assert_called_with(
            "Authorization", "Basic dXNlcjpwYXNzd29yZA=="
        )

    def test_wiki_it_requires_environment_configuration(self) -> None:
        with (
            patch.object(agent_api, "wiki_it", None),
            patch.dict(agent_api.os.environ, {}, clear=True),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "WIKI_IT_URL, WIKI_IT_USER, WIKI_IT_PASSWORD must be set",
            ):
                agent_api.get_wiki_it()

    def test_dispatches_wiki_page_requests(self) -> None:
        get_wiki_page = Mock(return_value="Wiki content")

        with patch.dict(
            agent_api.TOOL_HANDLERS,
            {"get_wiki_page": lambda args: get_wiki_page(args["page"])},
        ):
            result = agent_api.execute_tool("get_wiki_page", {"page": "start"})

        self.assertEqual(result, "Wiki content")
        get_wiki_page.assert_called_once_with("start")

    def test_rejects_unknown_tools(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown tool: missing"):
            agent_api.execute_tool("missing", {})


class WebSearchReActTests(unittest.TestCase):
    def test_repeats_web_search_until_the_llm_returns_an_answer(self) -> None:
        first_call = {
            "id": "call-1",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "first query"}',
            },
        }
        second_call = {
            "id": "call-2",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "second query"}',
            },
        }
        initial_message = {"role": "assistant", "tool_calls": [first_call]}
        follow_up = {
            "choices": [{"message": {"role": "assistant", "tool_calls": [second_call]}}]
        }
        final_response = {
            "choices": [{"message": {"role": "assistant", "content": "Final answer"}}]
        }

        with (
            patch("agent_api.execute_tool", side_effect=["first result", "second result"]) as execute_tool,
            patch("agent_api.call_llm", side_effect=[follow_up, final_response]) as call_llm,
        ):
            answer, results = agent_api.run_web_search_react(
                [{"role": "user", "content": "Question"}], initial_message
            )

        self.assertEqual(answer, "Final answer")
        self.assertEqual(results, ["first result", "second result"])
        self.assertEqual(execute_tool.call_count, 2)
        self.assertEqual(
            call_llm.call_args_list[0].kwargs["tools"],
            agent_api.WEB_SEARCH_TOOL_LIST,
        )
        self.assertEqual(
            call_llm.call_args_list[1].kwargs["tools"],
            agent_api.WEB_SEARCH_TOOL_LIST,
        )

    def test_limits_web_search_to_the_configured_maximum(self) -> None:
        tool_call = {
            "id": "call-1",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "query"}',
            },
        }
        tool_message = {"role": "assistant", "tool_calls": [tool_call]}
        repeated_response = {"choices": [{"message": tool_message}]}
        final_response = {
            "choices": [{"message": {"role": "assistant", "content": "Final answer"}}]
        }

        with (
            patch("agent_api.execute_tool", return_value="result") as execute_tool,
            patch(
                "agent_api.call_llm",
                side_effect=[repeated_response] * (agent_api.MAX_WEB_SEARCH_STEPS - 1)
                + [final_response],
            ) as call_llm,
        ):
            answer, results = agent_api.run_web_search_react(
                [{"role": "user", "content": "Question"}], tool_message
            )

        self.assertEqual(answer, "Final answer")
        self.assertEqual(len(results), agent_api.MAX_WEB_SEARCH_STEPS)
        self.assertEqual(execute_tool.call_count, agent_api.MAX_WEB_SEARCH_STEPS)
        self.assertNotIn(
            "tools", call_llm.call_args_list[-1].kwargs
        )


if __name__ == "__main__":
    unittest.main()
