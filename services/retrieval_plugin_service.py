from models.retrieval_plugin_query_models import Queries, ResultModel

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

    async def queries(self, queries: Queries) -> ResultModel:
        url = f"{self._url}/query"
        resp = await self._httpx_session.get(url, headers=self._headers, json=queries.dict())
        resp_data = resp.json()
        return ResultModel(**resp_data)
