import logging
import os
import re

import requests


logger = logging.getLogger(__name__)


DEFAULT_RAZOR_EDGE_QUERY_URL = "https://razor-edge.net/gold/razor_edge/query"
DEFAULT_ANDY_USERS_INDEX_URL = "https://razor-edge.net/gold/users/index"
DEFAULT_GOLD_LOGIN_URL = "https://razor-edge.net/gold/users/login"

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "query_razor_edge",
        "description": (
            "Execute a read-only SQL query against razor_edge via "
            "https://razor-edge.net/gold/razor_edge/query. The query argument must be SQL "
            "and must start with SELECT or WITH. Never pass natural language."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Read-only SQL only. Must start with SELECT or WITH. Example: "
                        "SELECT * FROM tags LIMIT 10"
                    ),
                }
            },
            "required": ["query"],
        },
    },
}


def _is_read_only_query(query: str) -> bool:
    normalized = query.strip().lower()
    if not normalized:
        return False
    return normalized.startswith("select") or normalized.startswith("with")


def _build_bearer_header(token: str) -> str:
    normalized = token.strip()
    if normalized.lower().startswith("bearer "):
        return normalized
    return f"Bearer {normalized}"


def _build_api_key_headers() -> dict[str, str]:
    api_key = (
        os.environ.get("RAZOR_EDGE_QUERY_API_KEY")
        or os.environ.get("ANDY_API_KEY")
        or os.environ.get("RAZOR_EDGE_API_KEY")
    )
    if not api_key:
        return {}
    return {"x-api-key": api_key.strip()}


def _without_api_key_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in {"x-api-key", "api-key"}
    }


def _extract_token_from_response(response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        for key in ("token", "access_token", "auth_token", "id_token", "jwt"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("token", "access_token", "auth_token", "id_token", "jwt"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    cookies = getattr(response, "cookies", None)
    if cookies is not None and hasattr(cookies, "items"):
        for cookie_name, cookie_value in cookies.items():
            if "token" in cookie_name.lower() and cookie_value:
                return cookie_value.strip()

    text = (getattr(response, "text", "") or "")
    patterns = (
        r'"(?:access_token|auth_token|id_token|jwt|token)"\s*:\s*"([^"]+)"',
        r"(?:access_token|auth_token|id_token|jwt|token)\s*=\s*['\"]([^'\"]+)['\"]",
        r"name=['\"](?:access_token|auth_token|id_token|jwt|token)['\"]\s+value=['\"]([^'\"]+)['\"]",
        r"Bearer\s+([A-Za-z0-9._-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def _fetch_andy_token(andy_user: str, andy_password: str, headers: dict[str, str]) -> str | None:
    response = requests.get(
        DEFAULT_ANDY_USERS_INDEX_URL,
        auth=(andy_user, andy_password),
        headers=headers,
        timeout=30,
        allow_redirects=True,
    )
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and status_code >= 400:
        logger.warning(
            "users/index returned status %s while fetching ANDY token; falling back",
            status_code,
        )
        return None
    return _extract_token_from_response(response)


def _resolve_app_credentials() -> tuple[str | None, str | None]:
    user = os.environ.get("ANDY_USER") or os.environ.get("RAZOR_EDGE_APP_USER")
    password = (
        os.environ.get("ANDY_PASSWORD")
        or os.environ.get("ANDY_PASSWORD_")
        or os.environ.get("RAZOR_EDGE_APP_PASSWORD")
        or os.environ.get("RAZOR_EDGE_APP_PASSWORD_")
    )
    return user, password


def _resolve_apache_basic_credentials() -> tuple[str | None, str | None]:
    user = (
        os.environ.get("BASIC_AUTHORIZATION_USER")
        or os.environ.get("APACHE_BASIC_USER")
        or os.environ.get("RAZOR_EDGE_BASIC_USER")
    )
    password = (
        os.environ.get("BASIC_AUTHORIZATION_PASSWORD")
        or os.environ.get("BASIC_AUTHORIZATION_PASSWORD_")
        or os.environ.get("APACHE_BASIC_PASSWORD")
        or os.environ.get("APACHE_BASIC_PASSWORD_")
        or os.environ.get("RAZOR_EDGE_BASIC_PASSWORD")
        or os.environ.get("RAZOR_EDGE_BASIC_PASSWORD_")
    )
    return user, password


def _query_with_apache_and_cakephp_session(
    query: str,
    headers: dict[str, str],
    apache_basic: tuple[str, str],
    app_user: str,
    app_password: str,
):
    session = requests.Session()
    session.headers.update(headers)
    session.auth = apache_basic

    login_response = session.post(
        DEFAULT_GOLD_LOGIN_URL,
        data={
            "data[User][username]": app_user,
            "data[User][password]": app_password,
        },
        timeout=30,
        allow_redirects=True,
    )

    if login_response.status_code >= 400:
        logger.warning(
            "CakePHP login returned status %s; falling back",
            login_response.status_code,
        )
        return None

    query_response = session.get(
        DEFAULT_RAZOR_EDGE_QUERY_URL,
        params={"query": query},
        timeout=30,
    )
    return query_response


def query_razor_edge(query: str):
    if not _is_read_only_query(query):
        raise ValueError("Only read-only queries are allowed (SELECT / WITH).")

    base_headers = _build_api_key_headers()
    andy_user = os.environ.get("RAZOR_EDGE_QUERY_USER")
    andy_password = (
        os.environ.get("RAZOR_EDGE_QUERY_PASSWORD")
        or os.environ.get("RAZOR_EDGE_QUERY_PASSWORD_")
    )
    app_user, app_password = _resolve_app_credentials()
    apache_basic_user, apache_basic_password = _resolve_apache_basic_credentials()
    andy_token = None

    if (
        apache_basic_user
        and apache_basic_password
        and app_user
        and app_password
    ):
        try:
            response = _query_with_apache_and_cakephp_session(
                query,
                headers=base_headers,
                apache_basic=(apache_basic_user, apache_basic_password),
                app_user=app_user,
                app_password=app_password,
            )
            if response is not None:
                if response.status_code != 401:
                    response.raise_for_status()
                    return response.json()
                logger.warning("query_razor_edge unauthorized with mode=apache_basic_plus_cakephp_session")
        except requests.RequestException as error:
            logger.warning("query_razor_edge session flow failed: %s", error)

    if andy_user and andy_password:
        try:
            andy_token = _fetch_andy_token(andy_user, andy_password, headers=base_headers)
        except Exception as error:
            # users/index can require a different auth mode; fall back to direct query auth.
            logger.warning("Failed to fetch ANDY token from users/index: %s", error)

    if not andy_token:
        andy_token = os.environ.get("RAZOR_EDGE_QUERY_TOKEN") or os.environ.get("ANDY_TOKEN")

    candidates: list[tuple[dict[str, str], tuple[str, str] | None, str]] = []
    seen: set[tuple[tuple[tuple[str, str], ...], tuple[str, str] | None]] = set()

    def add_candidate(headers: dict[str, str], auth: tuple[str, str] | None, label: str) -> None:
        key = (tuple(sorted(headers.items())), auth)
        if key in seen:
            return
        seen.add(key)
        candidates.append((headers, auth, label))

    if andy_token:
        add_candidate(
            {**base_headers, "Authorization": _build_bearer_header(andy_token)},
            None,
            "andy_token",
        )

    if andy_user and andy_password:
        add_candidate(dict(base_headers), (andy_user, andy_password), "andy_basic")

    # Backward-compatible fallback for environments still using bearer auth.
    authorization = os.environ.get("MONGODB_ACCESSOR_AUTHORIZATION")
    if authorization:
        add_candidate(
            {**base_headers, "Authorization": authorization},
            None,
            "legacy_authorization",
        )

    if not candidates:
        # Last-resort anonymous attempt for clearer diagnostics when no credentials are set.
        add_candidate(dict(base_headers), None, "no_auth")

    # Retry each candidate once without API key headers.
    for headers, auth, label in list(candidates):
        without_api_key = _without_api_key_headers(headers)
        if without_api_key != headers:
            add_candidate(without_api_key, auth, f"{label}_no_api_key")

    last_unauthorized_response = None
    last_exception: Exception | None = None

    for headers, auth, label in candidates:
        try:
            response = requests.get(
                DEFAULT_RAZOR_EDGE_QUERY_URL,
                params={"query": query},
                headers=headers,
                auth=auth,
                timeout=30,
            )
        except requests.RequestException as error:
            last_exception = error
            logger.warning("query_razor_edge request failed with mode=%s: %s", label, error)
            continue

        if response.status_code == 401:
            last_unauthorized_response = response
            logger.warning("query_razor_edge unauthorized with mode=%s", label)
            continue

        response.raise_for_status()
        return response.json()

    if last_unauthorized_response is not None:
        raise requests.HTTPError(
            "401 Unauthorized for razor_edge/query. Tried dual-auth and token/basic/legacy auth (with and without API key). Check BASIC_AUTHORIZATION_*, ANDY_* credentials, and API key scope.",
            response=last_unauthorized_response,
        )

    if last_exception is not None:
        raise last_exception

    raise RuntimeError("No authentication candidate was available for query_razor_edge")
