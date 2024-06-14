"""Main entrypoint for the app."""
import json
from typing import Optional
import openai
import uuid
import traceback
import os

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.templating import Jinja2Templates
from fastapi.responses import PlainTextResponse
from langchain.vectorstores import VectorStore
from websockets.exceptions import ConnectionClosedError
from langchain.docstore.document import Document
from services.aii_admin_service import AiiAdminApi
from services.retrieval_plugin_service import RetrievalPluginApi

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain, ChatHistoryVectorstoreQuerySupport, get_chat_history
from schemas import ChatResponse, LeadFormChatResponse
from utils.vectorstore_utils import EmptyVectorStore
from utils.prompt_utils import prompt_with_system_info
from utils.doc_links_in_answer_utils import doc_links_answer_support
from limits import ChatMessagesLimit
from models.retrieval_plugin_query_models import QueryResult as RetrievalPluginResult, Queries as RetrievalPluginQueries
from models.aii_admin_models import ChatSettings, ActionForCredits, LeadFormSettings, ChatUserTariffOpenAIKeySource
from logger import logger
from dependencies import http_dependencies
from utils.utils import is_disable_credit_mode_enabled
import random
from config import config


app = FastAPI()
# app.include_router(smart_seller_router)

templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None
general_openai_key = os.environ["OPENAI_API_KEY"]

# app.add_middleware(HTTPSRedirectMiddleware)
# app.mount("/.well-known/pki-validation", StaticFiles(directory="./.well-known/pki-validation"), name="static")


@app.on_event("startup")
async def startup():
    logger.info("#################### AII ####################")
    logger.info("START AII CHAT LANGCHAIN SERVICE")
    logger.info("#################### AII ####################")


@app.on_event("shutdown")
async def shutdown():
    await http_dependencies.httpx_session.aclose()


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/", response_class=PlainTextResponse)
async def get():
    return "Hello World"


def shuffle_string(s):
    shuffled_list = random.sample(s, len(s))
    return ''.join(shuffled_list)


def is_email_ai_answer(lead_form):
    return bool(lead_form.collect_lead_strategy == 'email_ai_answer')


@app.websocket("/trained_chat/{chat_id}/search/")
async def pre_trained_chat_search(
        websocket: WebSocket,
        chat_id: str,
        aii_admin_api: AiiAdminApi = Depends(http_dependencies.get_aii_admin_api),
        retrieval_plugin_api: RetrievalPluginApi = Depends(http_dependencies.get_retrieval_plugin_api)
):
    await websocket.accept()
    conn_id = str(uuid.uuid4())
    logger.info("connect#%s start chat_search#%s", conn_id, chat_id)
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
                search_query = req_data.get('search_query', None)
                ai_response_enabled = req_data.get('ai_response_enabled', True)

                clientVector = EmptyVectorStore()  # plug
                resp = ChatResponse(sender="you", message=search_query, type="stream")
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

                # make query to retrieval plugin
                queries = {
                  "queries": [
                    {
                      "query": search_query,
                      "filter": {
                          "author": f"chat_{chat_id}",
                      },
                      "top_k": chat_settings.langchain_chat_doc_count
                    }
                  ]
                }

                logger.info("connect#%s search query %s", conn_id, search_query)
                query_result: RetrievalPluginResult = await retrieval_plugin_api.query(
                    queries=RetrievalPluginQueries(**queries)
                )

                custom_docs = []
                for doc in query_result.results:
                    if chat_settings.score_vectorstore_docs_min_threshold and doc.score < chat_settings.score_vectorstore_docs_min_threshold:
                        continue
                    custom_docs.append(Document(page_content=doc.text, metadata={"url": doc.metadata.url}))

                logger.info("connect#%s search query founded docs count=%s", conn_id, len(custom_docs))
                docs_resp = ChatResponse(sender="bot", message=json.dumps(query_result.dict(exclude={"query"})), type="docs")
                await websocket.send_json(docs_resp.dict())

                if ai_response_enabled:
                    langchain_template = prompt_with_system_info(chat_settings.langchain_template)
                    qa_chain = get_chain(clientVector, question_handler, stream_handler,
                                         '', langchain_template,
                                         custom_docs=custom_docs, temperature=chat_settings.open_ai_temperature, model_name=chat_settings.model_name,
                                         top_k_docs_for_context=chat_settings.langchain_chat_doc_count)
                    result = await qa_chain.acall(
                       {"question": search_query, "chat_history": []}
                    )
                    chat_history.append((result['question'], result["answer"]))

                end_resp = ChatResponse(sender="bot", message="", type="end")
                await websocket.send_json(end_resp.dict())

                # increment user chat_messages_count per month
                incr_result = await aii_admin_api.increment_user_actions_count_per_month(chat_settings.owner.id)

            except WebSocketDisconnect:
                logger.info("connect#%s websocket disconnect", conn_id)
                break
    except Exception as e:
        logger.error(e, exc_info=True)
        resp = ChatResponse(
            sender="bot",
            message=str(e),
            type="error",
        )
        await websocket.send_json(resp.dict())
        logger.info("connect#%s websocket disconnect because of error", conn_id)


@app.websocket("/trained_chat/{chat_id}")
async def pre_trained_chat(
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


@app.websocket("/chat/v2/lead_form/{form_id}")
async def lead_form_chat_endpoint_v2(
        websocket: WebSocket,
        form_id,
        session_id: str = Query(None, description="Optional session ID for update filling"),
        filling_id: str = Query(None, description="Optional filling ID for update filling"),
        aii_admin_api: AiiAdminApi = Depends(http_dependencies.get_aii_admin_api),
):
    await websocket.accept()
    conn_id = str(uuid.uuid4())
    try:
        logger.info("connect#%s start form#%s chatting", conn_id, form_id)
        owner = await aii_admin_api.get_user_by_leadform_id(form_id)
        lead_form: LeadFormSettings = await aii_admin_api.get_lead_form(form_id)

        logger.debug("connect#%s owner %s", conn_id, owner)
        api_key = general_openai_key if owner['tariff']['open_ai_key_source'] == ChatUserTariffOpenAIKeySource.GENERAL.value else owner['openai_key']
        chat_messages_per_month_limit = owner['tariff']["chat_messages_per_month"] if owner['tariff'] else None
        logger.debug("connect#%s chat_messages_per_month_limit=%s", conn_id, chat_messages_per_month_limit)
        if not api_key:
            resp = LeadFormChatResponse(
                sender="bot",
                message="User with this form doesn't have api_key in aii_admin",
                type="error",
            )
            await websocket.send_json(resp.dict())
            logger.info("connect#%s user with this form#%s doesn't have api_key in aii_admin", conn_id, form_id)
            raise ConnectionClosedError

        while True:
            data = await websocket.receive_text()
            user_input = json.loads(data)["text"]
            logger.info('connect#%s user input: "%s"', conn_id, user_input)

            # check limits
            if chat_messages_per_month_limit is not None:
                user_actions_count_per_month = await aii_admin_api.get_user_actions_count_per_month(owner['id'])
                if user_actions_count_per_month.chat_messages_count >= chat_messages_per_month_limit:
                    resp = ChatResponse(sender="bot",
                                        message=f"Month limit messages {chat_messages_per_month_limit} exceeded",
                                        type="tariff_limit_exceeded")
                    await websocket.send_json(resp.dict())
                    logger.warning("connect#%s user#%s month limit messages exceed ws disconnect", conn_id, owner['id'])
                    raise WebSocketDisconnect

            # check credits
            if not is_disable_credit_mode_enabled(user_input):
                is_possible_to_spend_credits = await aii_admin_api.is_possible_to_spend_credits(owner['id'], action=ActionForCredits.AI_REPLY_LEAD_FORM.value)
                if not is_possible_to_spend_credits:
                    logger.warning("connect#%s user#%s month limit credits exceed ws disconnect", conn_id, owner['id'])
                    resp = ChatResponse(sender="bot",
                                        message=f"Month limit credits {chat_messages_per_month_limit} exceeded",
                                        type="tariff_limit_exceeded")
                    await websocket.send_json(resp.dict())
                    raise WebSocketDisconnect

            # Call GPT-3.5 API
            openai.api_key = api_key

            response = await openai.ChatCompletion.acreate(
                model=lead_form.model_name,
                messages=[{"role": "user", "content": user_input}],
                max_tokens=1500,
                n=1,
                stop=None,
                temperature=0.4,
                stream=True
            )

            # Extract the text from the response
            openai_resp = ''
            is_email_ai_answer_delimiter_sent = False
            try:
                start_resp = LeadFormChatResponse(sender="bot", message="", type="start")
                await websocket.send_json(start_resp.dict())
                async for response_chunk in response:
                    delta = response_chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        openai_resp += content

                        if is_email_ai_answer(lead_form) and len(openai_resp) > config.email_ai_answer_non_obfuscate_symbols:
                            if not is_email_ai_answer_delimiter_sent:
                                chat_resp = LeadFormChatResponse(
                                    sender="bot",
                                    message="email_ai_answer_delimiter",
                                    type="info",
                                )
                                await websocket.send_json(chat_resp.dict())
                                is_email_ai_answer_delimiter_sent = True

                            content = shuffle_string(content)

                        chat_resp = LeadFormChatResponse(
                            sender="bot",
                            message=content,
                            type="stream",
                        )
                        await websocket.send_json(chat_resp.dict())
                logger.debug('connect#%s openai response: "%s"', conn_id, openai_resp)
                end_resp = LeadFormChatResponse(sender="bot", message="", type="end")
                await websocket.send_json(end_resp.dict())
            finally:
                # increment user chat_messages_count per month
                incr_result = await aii_admin_api.increment_user_actions_count_per_month(owner['id'])
                logger.info(incr_result)
                if not is_disable_credit_mode_enabled(user_input):
                    await aii_admin_api.spend_credits(user_id=owner['id'], action=ActionForCredits.AI_REPLY_LEAD_FORM.value)

                if session_id and filling_id and is_email_ai_answer(lead_form):
                    logger.debug('connect#%s is_email_ai_answer and session_id and filling_id exist, try to update filling', conn_id)
                    await aii_admin_api.update_filling(form_id=form_id, session_id=session_id, filling_id=filling_id, ai_response=openai_resp)

    except (ConnectionClosedError, WebSocketDisconnect):
        await websocket.close()
        logger.info("connect#%s closed because of ConnectionClosedError", conn_id)
    except openai.OpenAIError as e:
        end_resp = LeadFormChatResponse(sender="bot", message=str(e), type="error")
        await websocket.send_json(end_resp.dict())
        await websocket.close()
        logger.error("connect#%s error occurred in WebSocket: %s\n%s", conn_id, e, traceback.format_exc())
    except Exception as e:
        await websocket.close()
        logger.error("connect#%s error occurred in WebSocket: %s\n%s", conn_id, e, traceback.format_exc())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
