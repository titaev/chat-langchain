import httpx

from config import config


class AiiAdminApi:
    _url = config.aii_admin_url
    _headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.aii_admin_secret_key}",
    }

    def __init__(self, httpx_session):
        self._httpx_session = httpx_session

    async def get_openai_key_by_leadform_id(self, form_id):
        url = f"{self._url}/api/v1/lead_forms/{form_id}/chatapi/user/"
        resp = await self._httpx_session.get(url, headers=self._headers)
        resp_data = resp.json()
        return resp_data['owner']['openai_key']

