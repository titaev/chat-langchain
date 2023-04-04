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


def _get_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    buffer = ""
    for human_s, ai_s in chat_history:
        human = "Human: " + human_s
        ai = "Assistant: " + ai_s
        buffer += "\n" + "\n".join([human, ai])
    return buffer


class MyChatVectorDBChain(ChatVectorDBChain):
    custom_results: Optional[str] = None

    def __init__(self, custom_results, **kwargs):
        super().__init__(**kwargs)  # вызываем конструктор родительского класса
        self.custom_results = custom_results

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
        if self.custom_results:
            doc = Document(
                page_content=self.custom_results
            )
            docs = [doc]
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
    vectorstore: VectorStore, question_handler, stream_handler, condense_template, qa_prompt, tracing: bool = False, custom_results=None
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
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = OpenAIChat(streaming=True, callback_manager=stream_manager, verbose=True, temperature=0, max_tokens=675)
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)
    QA_PROMPT = PromptTemplate(template=qa_prompt, input_variables=["question", "context"])
    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    )

    qa = MyChatVectorDBChain(
        custom_results,
        vectorstore=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        top_k_docs_for_context=4,
    )
    return qa
