from pydantic import BaseModel, validator
from typing import Optional


class ChatUserTariff(BaseModel):
    id: int
    name: str
    chat_messages_per_month: Optional[int]


class ChatUser(BaseModel):
    id: int
    email: str
    tariff: Optional[ChatUserTariff]


class ChatSettings(BaseModel):
    id: str
    name: str
    model_name: str
    open_ai_temperature: float
    langchain_template: str
    langchain_condense_template: str
    langchain_chat_doc_count: int
    owner: ChatUser

    @validator('langchain_template', 'langchain_condense_template', pre=True, always=True)
    def empty_string_if_none(cls, v):
        return '' if v is None else v


class UserActionsCountPerMonth(BaseModel):
    id: int
    year: int
    month: int
    chat_messages_count: int
    user: int
