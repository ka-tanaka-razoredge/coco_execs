# tools/web_search.py
from duckduckgo_search import DDGS

def search_web(query: str, max_results: int = 5):
    """
    DuckDuckGoを使ってWeb検索を行い、結果をリストで返します。
    """
    try:
        with DDGS() as ddgs:
            # textメソッドでウェブ検索を実行
            results = list(ddgs.text(query, max_results=max_results))
            
            # LLMが読みやすいように、タイトル、URL、スニペット（概要）を整理
            formatted_results = [
                {
                    "title": r.get("title"),
                    "href": r.get("href"),
                    "body": r.get("body")
                }
                for r in results
            ]
            return {"results": formatted_results}
            
    except Exception as e:
        return {"error": f"Failed to execute web search: {str(e)}"}
