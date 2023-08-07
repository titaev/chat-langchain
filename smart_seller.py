from fastapi import APIRouter, Depends

import uuid

import json

from fastapi import WebSocket, WebSocketDisconnect
from langchain.docstore.document import Document

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain, ChatHistoryVectorstoreQuerySupport, get_chat_history
from schemas import ChatResponse
from services.aii_admin_service import AiiAdminApi
from services.retrieval_plugin_service import RetrievalPluginApi
from utils.vectorstore_utils import EmptyVectorStore
from utils.prompt_utils import prompt_with_system_info
from utils.doc_links_in_answer_utils import doc_links_answer_support
from limits import ChatMessagesLimit
from models.retrieval_plugin_query_models import QueryResult as RetrievalPluginResult, Queries as RetrievalPluginQueries
from models.aii_admin_models import ChatSettings
from logger import logger
from dependencies import http_dependencies

router = APIRouter()


@router.websocket("/trained_chat/smart_seller/{chat_id}")
async def smart_seller_router(
        websocket: WebSocket,
        chat_id: str,
        aii_admin_api: AiiAdminApi = Depends(http_dependencies.get_aii_admin_api),
        retrieval_plugin_api: RetrievalPluginApi = Depends(http_dependencies.get_retrieval_plugin_api)
):
    await websocket.accept()
    conn_id = str(uuid.uuid4())
    logger.info("connect#%s start chat#%s", conn_id, chat_id)
    try:
        question_handler = QuestionGenCallbackHandler(websocket)
        stream_handler = StreamingLLMCallbackHandler(websocket)
        chat_history = []
        chat_settings: ChatSettings = await aii_admin_api.get_chat(chat_id)
        logger.debug("connect#%s chat_settings#%s", conn_id, chat_settings)
        while True:
            try:
                # Receive and send back the client message
                request = await websocket.receive_text()
                req_data = json.loads(request)
                messages = req_data.get('messages', None)
                question = messages[-1]['content']
                logger.info("connect#%s chat question %s", conn_id, question)

                # if not(user):
                #     resp = ChatResponse(
                #         sender="bot",
                #         message="Client with such id doesn't exist.",
                #         type="error",
                #     )
                #     await websocket.send_json(resp.dict())
                #     continue

                clientVector = EmptyVectorStore()  # plug
                resp = ChatResponse(sender="you", message=question, type="stream")
                await websocket.send_json(resp.dict())

                # check limits
                chat_messages_limit = ChatMessagesLimit(chat_settings, conn_id, aii_admin_api)
                if not await chat_messages_limit.is_chat_message_allowed():
                    resp = ChatResponse(sender="bot", message=f"Month limit messages {chat_messages_limit.per_month} exceeded", type="tariff_limit_exceeded")
                    await websocket.send_json(resp.dict())
                    raise WebSocketDisconnect

                # Construct a response
                start_resp = ChatResponse(sender="bot", message="", type="start")
                await websocket.send_json(start_resp.dict())

                # form new question chat_history_support
                lim_chat_history = messages[-8:-1] if messages else []
                if chat_settings.langchain_chat_history_enable:
                    logger.debug("connect#%s chat history enabled", conn_id)
                    logger.debug("connect#%s chat history %s", conn_id, lim_chat_history)
                    logger.debug("connect#%s user question %s", conn_id, question)
                    if lim_chat_history:
                        chat_history_support = ChatHistoryVectorstoreQuerySupport(question_handler, condense_template=chat_settings.langchain_condense_template)
                        question = await chat_history_support.get_new_question(question, lim_chat_history)
                        logger.debug("connect#%s created question %s", conn_id, question)

                # make query to retrieval plugin
                queries = {
                  "queries": [
                    {
                      "query": question,
                      "filter": {
                          "author": f"chat_{chat_id}",
                      },
                      "top_k": chat_settings.langchain_chat_doc_count
                    }
                  ]
                }
                query_result: RetrievalPluginResult = await retrieval_plugin_api.query(
                    queries=RetrievalPluginQueries(**queries)
                )

                custom_docs = []
                for doc in query_result.results:
                    if chat_settings.score_vectorstore_docs_min_threshold and doc.score < chat_settings.score_vectorstore_docs_min_threshold:
                        continue
                    custom_docs.append(Document(page_content=doc.text, metadata={"url": doc.metadata.url}))

                logger.info("connect#%s search query founded docs (with appropriate score) count=%s", conn_id, len(custom_docs))
                if chat_settings.doc_links_in_answer_enabled:
                    custom_docs = doc_links_answer_support.enrich_docs_with_source_links(custom_docs)
                logger.info("connect#%s search query founded docs custom_docs=%s", conn_id, custom_docs)

                # send source links
                if chat_settings.references_enabled:
                    source_links = [doc.metadata.url for doc in query_result.results]
                    docs_resp = ChatResponse(sender="bot", message=json.dumps(source_links), type="source_links")
                    await websocket.send_json(docs_resp.dict())

                langchain_template = chat_settings.langchain_template
                if chat_settings.doc_links_in_answer_enabled:
                    langchain_template = doc_links_answer_support.prompt_support(langchain_template)
                langchain_template = prompt_with_system_info(langchain_template, langchain_chat_history_prompt_enable=chat_settings.langchain_chat_history_prompt_enable)
                logger.info("connect#%s prompt=%s", conn_id, langchain_template)

                qa_chain = get_chain(clientVector, question_handler, stream_handler,
                                     '', langchain_template,
                                     custom_docs=custom_docs, temperature=chat_settings.open_ai_temperature, model_name=chat_settings.model_name,
                                     top_k_docs_for_context=chat_settings.langchain_chat_doc_count, langchain_chat_history_prompt_enable=chat_settings.langchain_chat_history_prompt_enable)

                lim_chat_history_str = ''
                if chat_settings.langchain_chat_history_prompt_enable:
                    lim_chat_history_str = get_chat_history(lim_chat_history)

                result = await qa_chain.acall(
                   {"question": question, "chat_history": lim_chat_history_str}
                )
                chat_history.append((result['question'], result["answer"]))
                logger.info("connect#%s, openai answer %s", conn_id, result["answer"])


                end_resp = ChatResponse(sender="bot", message="", type="end")
                await websocket.send_json(end_resp.dict())

                # increment user chat_messages_count per month
                incr_result = await aii_admin_api.increment_user_actions_count_per_month(chat_settings.owner.id)
                logger.info("connect#%s, %s", conn_id, incr_result)

            except WebSocketDisconnect:
                logger.info("connect#%s websocket disconnect", conn_id)
                break
    except Exception as e:
        logger.error(e, exc_info=True)
        logger.info("connect#%s websocket disconnect because of error", conn_id)
        resp = ChatResponse(
            sender="bot",
            message=str(e),
            type="error",
        )
        await websocket.send_json(resp.dict())
