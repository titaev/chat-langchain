from pydantic import BaseModel, validator


class ChatSettings(BaseModel):
    id: str
    name: str
    model_name: str
    langchain_template: str
    langchain_condense_template: str
    langchain_chat_doc_count: int
    owner: int

    @validator('langchain_template', 'langchain_condense_template', pre=True, always=True)
    def empty_string_if_none(cls, v):
        return '' if v is None else v
