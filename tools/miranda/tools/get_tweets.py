import os

import requests


DEFAULT_TWEETS_URL = "https://razor-edge.net:3110/tweets"
DEFAULT_TWEETS_LOGIN_URL = "https://razor-edge.net:3110/auth/login"
TWEETS_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_tweets",
        "description": (
            "Log in to https://razor-edge.net:3110/auth/login using configured credentials "
            "to obtain a fresh token, then fetch tweets for answering questions about their "
            "contents. Defaults to page 1 with up to 100 tweets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Positive integer. Numeric strings are also accepted.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 100,
                    "description": "Positive integer. Numeric strings are also accepted.",
                },
            },
        },
    },
}


def _positive_integer(value: int | str, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, str):
        value = value.strip()
        if not value.isdecimal():
            raise ValueError(f"{name} must be a positive integer")
    elif not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _get_login_credentials() -> tuple[str, str]:
    username = os.environ.get("TWEETS_API_USERNAME") or os.environ.get("ANDY_USER")
    password = os.environ.get("TWEETS_API_PASSWORD") or os.environ.get("ANDY_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "TWEETS_API_USERNAME and TWEETS_API_PASSWORD (or ANDY_USER and ANDY_PASSWORD) "
            "must be set to fetch tweets"
        )
    return username, password


def _get_tweets_token() -> str:
    username, password = _get_login_credentials()
    response = requests.post(
        DEFAULT_TWEETS_LOGIN_URL,
        json={"username": username, "password": password},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    token = None
    if isinstance(payload, dict):
        for key in ("token", "access_token", "auth_token", "jwt"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                token = value.strip()
                break
        if token is None and isinstance(payload.get("data"), dict):
            for key in ("token", "access_token", "auth_token", "jwt"):
                value = payload["data"].get(key)
                if isinstance(value, str) and value.strip():
                    token = value.strip()
                    break

    if not token:
        raise ValueError("Tweet login response did not contain a token")
    if token.lower().startswith("bearer "):
        return token
    return f"Bearer {token}"


def get_tweets(page: int | str = 1, limit: int | str = 100):
    page = _positive_integer(page, "page")
    limit = _positive_integer(limit, "limit")
    token = _get_tweets_token()

    response = requests.get(
        DEFAULT_TWEETS_URL,
        params={"page": page, "limit": limit},
        headers={"Authorization": token},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()
