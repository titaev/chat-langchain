"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAIChat
from langchain.vectorstores.base import VectorStore
from langchain.prompts.prompt import PromptTemplate
from typing import Any, Dict, List, Tuple, Optional
from langchain.docstore.document import Document
from logger import logger


def _get_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    buffer = ""
    for human_s, ai_s in chat_history:
        human = "Human: " + human_s
        ai = "Assistant: " + ai_s
        buffer += "\n" + "\n".join([human, ai])
    return buffer


class ChatHistorySupport:
    default_condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. 
    This question must be in the same language as Follow up question. Follow up question has a greater advantage in meaning, because this is the last question from the human in the conversation. But it is also worth considering the entire history of the conversation. 
    You should assume that this should be question. If Follow up question is very different from the previous conversation in meaning, then just return this Follow up question as a result. Do not try to take the result from the user's questions directly from the Chat History.
                            Chat History:
                            {chat_history}
                            Follow up question: {question}
                            Standalone question:"""

    def __init__(self, question_handler, model_name='gpt-3.5-turbo', condense_template=None):
        condense_template = condense_template if condense_template else self.default_condense_template
        manager = AsyncCallbackManager([])
        question_manager = AsyncCallbackManager([question_handler])
        condence_question_prompt = PromptTemplate.from_template(condense_template)
        question_gen_llm = OpenAIChat(
            model_name=model_name,
            temperature=0.1,
            verbose=True,
            callback_manager=question_manager,
        )
        self.question_generator = LLMChain(
            llm=question_gen_llm, prompt=condence_question_prompt, callback_manager=manager
        )

    async def get_new_question(self, question, chat_history: List[Dict]):
        """create new question from chat history and current question"""
        chat_history_str = self._get_chat_history(chat_history)
        if chat_history_str:
            new_question = await self.question_generator.arun(
                question=question, chat_history=chat_history_str
            )
            logger.debug("chat_history_str %s", chat_history_str)
        else:
            new_question = question
        return new_question

    @staticmethod
    def _get_chat_history(chat_history: List[Dict]) -> str:
        buffer = ""
        for msg in chat_history:
            if msg["role"] == "user":
                buffer_msg = f"Human: {msg['content']}"
            else:
                buffer_msg = f"Assistant: {msg['content']}"
            buffer += f"\n {buffer_msg}"
        return buffer


class MyChatVectorDBChain(ChatVectorDBChain):
    custom_docs: List[Document] = None

    def __init__(self, custom_docs, **kwargs):
        super().__init__(**kwargs)  # вызываем конструктор родительского класса
        self.custom_docs = custom_docs

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        chat_history_str = _get_chat_history(inputs["chat_history"])
        vectordbkwargs = inputs.get("vectordbkwargs", {})
        if chat_history_str:
            new_question = await self.question_generator.arun(
                question=question, chat_history=chat_history_str
            )
        else:
            new_question = question
        # TODO: This blocks the event loop, but it's not clear how to avoid it.
        if self.custom_docs:
            docs = self.custom_docs
        else:
            docs = self.vectorstore.similarity_search(
                new_question, k=self.top_k_docs_for_context, **vectordbkwargs
            )
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        answer, _ = await self.combine_docs_chain.acombine_docs(docs, **new_inputs)
        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}


def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler,
        condense_template, qa_prompt, tracing: bool = False, custom_docs=None,
        temperature=0, model_name='gpt-3.5-turbo', top_k_docs_for_context=4
) -> ChatVectorDBChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = OpenAIChat(
        model_name=model_name,
        temperature=temperature,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = OpenAIChat(
        model_name=model_name,
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=temperature,
        max_tokens=-1   # no limit (openai api max limit)
    )
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)
    QA_PROMPT = PromptTemplate(template=qa_prompt, input_variables=["question", "context"])
    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    )

    qa = MyChatVectorDBChain(
        custom_docs,
        vectorstore=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        top_k_docs_for_context=top_k_docs_for_context,
    )
    return qa
