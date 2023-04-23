import httpx

from config import config


class AiiAdminApi:
    _url = config.aii_admin_url
    _token = config.aii_admin_secret_key

    def __init__(self, httpx_session):
        self._httpx_session = httpx_session

    async def get_user_by_leadform_id(self, form_id):
        pass
