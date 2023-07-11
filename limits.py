from logger import logger


class ChatMessagesLimit:
    def __init__(self, chat_settings, ws_conn_id, aii_admin_api):
        self.aii_admin_api = aii_admin_api
        self.ws_conn_id = ws_conn_id
        self.chat_settings = chat_settings

        self.per_month = chat_settings.owner.tariff.chat_messages_per_month if chat_settings.owner.tariff else None
        logger.debug("connect#%s chat_messages_per_month_limit=%s", ws_conn_id, self.per_month)

    async def is_chat_message_allowed(self):
        if self.per_month is None:
            return True

        user_actions_count_per_month = await self.aii_admin_api.get_user_actions_count_per_month(self.chat_settings.owner.id)
        if user_actions_count_per_month.chat_messages_count >= self.per_month:
            logger.warning("connect#%s user#%s month limit messages exceed", self.ws_conn_id, self.chat_settings.owner.id)
            return False

        return True

