import os
from base64 import b64encode
from xmlrpc.client import ServerProxy, Transport


DEFAULT_WIKI_DO_URL = "https://razor-edge.net/study/do/dokuwiki_observee/lib/exe/xmlrpc.php"


class BasicAuthTransport(Transport):
    def __init__(self, user: str, password: str):
        super().__init__()
        credentials = f"{user}:{password}".encode("utf-8")
        self.authorization = f"Basic {b64encode(credentials).decode('ascii')}"

    def send_headers(self, connection, headers):
        super().send_headers(connection, headers)
        connection.putheader("Authorization", self.authorization)


class WikiDo:
    def __init__(self, url: str, user: str, password: str):
        if not url:
            raise ValueError("url must not be empty")
        if not user:
            raise ValueError("user must not be empty")
        if not password:
            raise ValueError("password must not be empty")

        self.url = url
        self.user = user
        self.password = password
        self.server = ServerProxy(
            url,
            transport=BasicAuthTransport(user, password),
            allow_none=True,
        )

        if not self.server.dokuwiki.login(user, password):
            raise RuntimeError("DokuWiki login failed")

    def get_page(self, page: str) -> str:
        if not page:
            raise ValueError("page must not be empty")
        return self.server.wiki.getPage(page)

    def put_page(self, page: str, text: str, summary: str = "") -> bool:
        if not page:
            raise ValueError("page must not be empty")
        if text is None:
            raise ValueError("text must not be None")
        options = {"sum": summary} if summary else {}
        return bool(self.server.wiki.putPage(page, text, options))

    def list_pages(self) -> list:
        return self.server.wiki.getAllPages()

    def search(self, keyword: str) -> list:
        if not keyword:
            raise ValueError("keyword must not be empty")
        return self.server.dokuwiki.search(keyword)


def create_wiki_do(
    url: str = DEFAULT_WIKI_DO_URL,
    user: str | None = None,
    password: str | None = None,
) -> WikiDo:
    resolved_user = user or os.environ.get("WIKI_DO_USER")
    resolved_password = password or os.environ.get("WIKI_DO_PASSWORD")

    if not resolved_user or not resolved_password:
        raise RuntimeError("Set WIKI_DO_USER/WIKI_DO_PASSWORD")

    return WikiDo(url=url, user=resolved_user, password=resolved_password)
