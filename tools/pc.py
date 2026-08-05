import requests

FASTAPI_SQL_URL = "http://192.168.10.8:1564/sql"

def call_pc(query: str):
    res = requests.get(
        FASTAPI_SQL_URL,
        params={"query": query}
    )
    return res.json()
