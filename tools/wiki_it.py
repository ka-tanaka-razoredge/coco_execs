from base64 import b64encode
from xmlrpc.client import ServerProxy, Transport


class BasicAuthTransport(Transport):
    def __init__(self, user, password):
        super().__init__()
        credentials = f"{user}:{password}".encode()
        self.authorization = f"Basic {b64encode(credentials).decode()}"

    def send_headers(self, connection, headers):
        super().send_headers(connection, headers)
        connection.putheader("Authorization", self.authorization)


class WikiIt:
    def __init__(self, url, user, password):
        self.server = ServerProxy(
            url,
            transport=BasicAuthTransport(user, password),
        )
        self.user = user
        self.password = password

        # 接続確認
        if not self.server.dokuwiki.login(user, password):
            raise Exception("DokuWiki login failed")

    def get_page(self, page):
        return self.server.wiki.getPage(page)

    # def put_page(self, page, text):
    #     return self.server.wiki.putPage(page, text, {})

    def list_pages(self):
        return self.server.wiki.getAllPages()

    def search(self, keyword):
        return self.server.dokuwiki.search(keyword)
