import httpx
from services.aii_admin_service import AiiAdminApi
from services.retrieval_plugin_service import RetrievalPluginApi


class HTTPDependencies:
    def __init__(self):
        self.httpx_session = httpx.AsyncClient()

    def get_httpx_session(self):
        return self.httpx_session

    def get_aii_admin_api(self):
        return AiiAdminApi(self.httpx_session)

    def get_retrieval_plugin_api(self):
        return RetrievalPluginApi(self.httpx_session)


http_dependencies = HTTPDependencies()
