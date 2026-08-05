import requests

FASTAPI_SQL_URL = "http://192.168.10.8:1564/sql"


def _is_read_only_query(query: str) -> bool:
    normalized = query.strip().lower()
    if not normalized:
        return False
    return normalized.startswith("select") or normalized.startswith("with")

def call_pc(query: str):
    if not _is_read_only_query(query):
        raise ValueError("Only read-only queries are allowed (SELECT / WITH).")

    res = requests.get(
        FASTAPI_SQL_URL,
        params={"query": query}
    )
    return res.json()
