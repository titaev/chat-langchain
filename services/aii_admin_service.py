from typing import Optional

from config import config
from models.aii_admin_models import ChatSettings, UserActionsCountPerMonth


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

    async def get_chat(self, chat_id) -> Optional[ChatSettings]:
        url = f"{self._url}/api/v1/aii_chat_api/training/chat/{chat_id}"
        resp = await self._httpx_session.get(url, headers=self._headers)
        resp_data = resp.json()
        return ChatSettings(**resp_data)

    async def get_user_actions_count_per_month(self, user_id) -> Optional[UserActionsCountPerMonth]:
        url = f"{self._url}/api/v1/aii_chat_api/user/{user_id}/actions_count/month/"
        resp = await self._httpx_session.get(url, headers=self._headers)
        resp_data = resp.json()
        return UserActionsCountPerMonth(**resp_data)

    async def increment_user_actions_count_per_month(self, user_id) -> Optional[UserActionsCountPerMonth]:
        url = f"{self._url}/api/v1/aii_chat_api/user/{user_id}/actions_count/month/chat_messages/increment/"
        data = {"count": 1}
        resp = await self._httpx_session.post(url, headers=self._headers, json=data)
        resp_data = resp.json()
        return resp_data
