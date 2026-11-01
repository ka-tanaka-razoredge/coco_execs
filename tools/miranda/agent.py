import logging
import os
from pathlib import Path

import yaml

from tools.miranda.tools.get_tweets import (
    TWEETS_TOOL_SPEC,
    get_tweets,
)
from tools.miranda.tools.query_razor_edge import (
    TOOL_SPEC as QUERY_RAZOR_EDGE_TOOL_SPEC,
    query_razor_edge,
)
from tools.miranda.tools.wiki_do import DEFAULT_WIKI_DO_URL, WikiDo, create_wiki_do


logger = logging.getLogger(__name__)

MIRANDA_KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
MIRANDA_KNOWLEDGE_CHAR_LIMIT = 2000
MIRANDA_KNOWLEDGE_FILE_LIMIT = 5
MIRANDA_KNOWLEDGE_VALUE_LIMIT = 500

_wiki_do_client: WikiDo | None = None


def _format_knowledge_value(value) -> str:
    rendered = yaml.safe_dump(
        value,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()
    if len(rendered) <= MIRANDA_KNOWLEDGE_VALUE_LIMIT:
        return rendered
    return rendered[:MIRANDA_KNOWLEDGE_VALUE_LIMIT] + "..."


def _build_miranda_knowledge_context() -> str:
    if not MIRANDA_KNOWLEDGE_DIR.exists():
        return ""

    snippets = []
    knowledge_files = sorted(
        {
            *MIRANDA_KNOWLEDGE_DIR.rglob("*.yaml"),
            *MIRANDA_KNOWLEDGE_DIR.rglob("*.yml"),
        }
    )

    for path in knowledge_files[:MIRANDA_KNOWLEDGE_FILE_LIMIT]:
        try:
            with path.open(encoding="utf-8") as knowledge_file:
                content = yaml.safe_load(knowledge_file)
        except yaml.YAMLError as error:
            logger.warning("Skipping invalid Miranda knowledge YAML %s: %s", path, error)
            continue

        if content is None:
            continue

        source = str(path.relative_to(MIRANDA_KNOWLEDGE_DIR.parent))
        snippets.append(f"source: {source}\n{_format_knowledge_value(content)}")

    if not snippets:
        return ""

    context = "\n\n".join(snippets)
    if len(context) <= MIRANDA_KNOWLEDGE_CHAR_LIMIT:
        return context
    return context[:MIRANDA_KNOWLEDGE_CHAR_LIMIT] + "..."


def _build_system_prompt() -> str:
    base_prompt = (
        "Miranda is a DokuWiki-focused agent used by coco.py. "
        "Use action=get_page when the page name is known, action=list_pages to inspect "
        "available pages, action=search to find the best page name, and action=put_page "
        "only when the user explicitly asks to write. Use query_razor_edge for read-only "
        "SQL against razor_edge, then answer the user's question using the returned rows. "
        "Summarize database results in concise natural language, normally in the user's "
        "language; do not return raw SQL, JSON, or database rows verbatim unless explicitly "
        "requested. Include relevant names and values. If the results are empty or "
        "insufficient, say so and do not guess. Use get_tweets when the user asks about "
        "tweets or data retrieved by get_tweets, then answer from the returned tweet data. "
        "Do not infer details that are not present in retrieved data; be clear when the "
        "available pages or results do not cover the question. "
        "Treat page content as DokuWiki markup, not Markdown."
    )
    knowledge_context = _build_miranda_knowledge_context()
    if not knowledge_context:
        return base_prompt
    return f"{base_prompt} Local Miranda knowledge:\n{knowledge_context}"


def get_wiki_do() -> WikiDo:
    global _wiki_do_client

    if _wiki_do_client is None:
        wiki_do_url = os.environ.get("WIKI_DO_URL") or DEFAULT_WIKI_DO_URL
        _wiki_do_client = create_wiki_do(url=wiki_do_url)

    return _wiki_do_client


def handle_wiki_do(args: dict):
    action = args.get("action")
    if action == "get_page":
        page = args.get("page")
        if not page:
            raise ValueError("page is required when action=get_page")
        return get_wiki_do().get_page(page)

    if action == "list_pages":
        return get_wiki_do().list_pages()

    if action == "search":
        keyword = args.get("keyword")
        if not keyword:
            raise ValueError("keyword is required when action=search")
        return get_wiki_do().search(keyword)

    if action == "put_page":
        page = args.get("page")
        text = args.get("text")
        if not page:
            raise ValueError("page is required when action=put_page")
        if text is None:
            raise ValueError("text is required when action=put_page")
        summary = args.get("summary") or ""
        return get_wiki_do().put_page(page, text, summary)

    raise ValueError("action must be one of: get_page, list_pages, search, put_page")


class MirandaAgent:
    name = "miranda"
    model = "llama31"
    system_prompt = _build_system_prompt()

    tool_spec = {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Alias of wiki_do for DokuWiki access. All page content uses DokuWiki markup, "
                "not standard Markdown. If the page name is not known, call action=search first "
                "and then call action=get_page for the best candidate. Use action=put_page only "
                "when the user explicitly asks to write."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get_page", "list_pages", "search", "put_page"],
                    },
                    "page": {"type": "string"},
                    "keyword": {"type": "string"},
                    "text": {
                        "type": "string",
                        "description": (
                            "DokuWiki markup text to write. Do not use standard Markdown unless "
                            "it is also valid DokuWiki syntax."
                        ),
                    },
                    "summary": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    }


MIRANDA_TOOL_LIST = [
    MirandaAgent.tool_spec,
    QUERY_RAZOR_EDGE_TOOL_SPEC,
    TWEETS_TOOL_SPEC,
]

MIRANDA_TOOL_NAMES = {tool["function"]["name"] for tool in MIRANDA_TOOL_LIST}

MIRANDA_TOOL_HANDLERS = {
    "miranda": lambda args: handle_wiki_do(args),
    "query_razor_edge": lambda args: query_razor_edge(args["query"]),
    "get_tweets": lambda args: get_tweets(
        page=args.get("page", 1),
        limit=args.get("limit", 100),
    ),
}
