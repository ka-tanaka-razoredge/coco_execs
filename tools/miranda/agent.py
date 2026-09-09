import logging
from pathlib import Path

import yaml


logger = logging.getLogger(__name__)

MIRANDA_KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
MIRANDA_KNOWLEDGE_CHAR_LIMIT = 2000
MIRANDA_KNOWLEDGE_FILE_LIMIT = 5
MIRANDA_KNOWLEDGE_VALUE_LIMIT = 500


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
        "SQL against razor_edge. Treat page content as DokuWiki markup, not Markdown."
    )
    knowledge_context = _build_miranda_knowledge_context()
    if not knowledge_context:
        return base_prompt
    return f"{base_prompt} Local Miranda knowledge:\n{knowledge_context}"


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
