# MongoDB Accessor API

FastAPIとMongoDBを接続する非同期CRUD APIです。

MongoDBを起動したうえで、Python 3.9以降を用意してください。

```bash
cp .env.example .env
python3.9 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 3800
```

APIは `http://localhost:3800`、Swagger UIは `http://localhost:3800/docs` です。

MongoDBが別ホストの場合は、`.env` の `MONGODB_URI` を接続先に変更します。

## agent_api.py の認証設定

`agent_api.py` は DokuWiki 接続に以下の環境変数を必要とします。`.env.example` を参照して実行環境に設定してください。`.env` は Git の追跡対象外です。

| Variable | Description |
| --- | --- |
| `WIKI_IT_URL` | DokuWiki XML-RPC endpoint |
| `WIKI_IT_USER` | DokuWiki user name |
| `WIKI_IT_PASSWORD` | DokuWiki password |
| `MONGODB_ACCESSOR_AUTHORIZATION` | MongoDB accessor API authorization header |

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | MongoDB接続状態を確認 |
| `POST` | `/{collection}` | ドキュメントを作成 |
| `GET` | `/{collection}?skip=0&limit=100` | ドキュメント一覧を取得 |
| `GET` | `/{collection}/{document_id}` | ドキュメントを取得 |
| `PATCH` | `/{collection}/{document_id}` | ドキュメントを更新 |
| `DELETE` | `/{collection}/{document_id}` | ドキュメントを削除 |

`{collection}` には `knowledge`、`history`、または `preferences` を指定します。作成リクエスト例:

```json
{
  "title": "MongoDB access",
  "content": "Knowledge document"
}
```
