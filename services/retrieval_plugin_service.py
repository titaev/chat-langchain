from typing import Optional
from models.retrieval_plugin_query_models import Queries, QueryResult

from config import config


class RetrievalPluginApi:
    """https://github.com/openai/chatgpt-retrieval-plugin"""
    _url = config.retrieval_plugin_url
    _headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.retrieval_plugin_secret_key}",
    }

    def __init__(self, httpx_session):
        self._httpx_session = httpx_session

    async def query(self, queries: Queries) -> Optional[QueryResult]:
        url = f"{self._url}/query"
        resp = await self._httpx_session.post(url, headers=self._headers, json=queries.dict())
        if resp.status_code == 200:
            print(resp.text)  # print the response text
            resp_data = resp.json()['results']
            return QueryResult(**resp_data[0]) if resp_data else None
        else:
            print(f"Error: {resp.status_code}")
            raise
