from pydantic import BaseModel, validator
from typing import Optional
from enum import Enum


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
    score_vectorstore_docs_min_threshold: Optional[float]
    langchain_template: str
    langchain_condense_template: str
    langchain_chat_doc_count: int
    langchain_chat_history_enable: bool
    langchain_chat_history_prompt_enable: bool
    owner: ChatUser
    references_enabled: bool
    doc_links_in_answer_enabled: bool

    @validator('langchain_template', 'langchain_condense_template', pre=True, always=True)
    def empty_string_if_none(cls, v):
        return '' if v is None else v


class UserActionsCountPerMonth(BaseModel):
    id: int
    year: int
    month: int
    chat_messages_count: int
    user: int


class ActionForCredits(Enum):
    AI_REPLY_LEAD_FORM = "ai_reply_lead_form"
    AI_CREATE_LEAD_FORM = "ai_create_lead_form"
