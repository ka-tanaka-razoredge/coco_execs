## tools

### query_pc

## coco.py の認証設定

`coco.py` は DokuWiki 接続に以下の環境変数を必要とします。`.env.example` を参照して実行環境に設定してください。`.env` は Git の追跡対象外です。

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

# else

## sitry

192.168.10.8:1564でaccessできるChroma accessor．
