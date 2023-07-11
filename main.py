"""Main entrypoint for the app."""
import json
import httpx
from typing import Optional
import openai
import uuid
import traceback
import os

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import PlainTextResponse
from langchain.vectorstores import VectorStore
from websockets.exceptions import ConnectionClosedError
from langchain.docstore.document import Document

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain, ChatHistorySupport
from schemas import ChatResponse, LeadFormChatResponse
from services.aii_admin_service import AiiAdminApi
from services.retrieval_plugin_service import RetrievalPluginApi
from utils.vectorstore_utils import EmptyVectorStore
from utils.prompt_utils import prompt_with_system_info
from utils.doc_links_in_answer_utils import doc_links_answer_support
from limits import ChatMessagesLimit
from models.retrieval_plugin_query_models import QueryResult as RetrievalPluginResult, Queries as RetrievalPluginQueries
from models.aii_admin_models import ChatSettings
from logger import logger

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None
httpx_session: httpx.AsyncClient
aii_admin_api: AiiAdminApi
retrieval_plugin_api: RetrievalPluginApi
general_openai_key = os.environ["OPENAI_API_KEY"]

# app.add_middleware(HTTPSRedirectMiddleware)
# app.mount("/.well-known/pki-validation", StaticFiles(directory="./.well-known/pki-validation"), name="static")


@app.on_event("startup")
async def startup():
    logger.info("#################### AII ####################")
    logger.info("START AII CHAT LANGCHAIN SERVICE")
    logger.info("#################### AII ####################")

    global httpx_session
    global aii_admin_api
    global retrieval_plugin_api

    httpx_session = httpx.AsyncClient()
    aii_admin_api = AiiAdminApi(httpx_session)
    retrieval_plugin_api = RetrievalPluginApi(httpx_session)


@app.on_event("shutdown")
async def shutdown():
    global httpx_session
    await httpx_session.aclose()


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/", response_class=PlainTextResponse)
async def get():
    return "Hello World"


@app.websocket("/trained_chat/{chat_id}/search/")
async def pre_trained_chat_search(websocket: WebSocket, chat_id: str):
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

                custom_docs = [Document(page_content=doc.text) for doc in query_result.results]
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
async def pre_trained_chat(websocket: WebSocket, chat_id: str):
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
                if chat_settings.langchain_chat_history_enable:
                    logger.debug("connect#%s chat history enabled", conn_id)
                    lim_chat_history = messages[-8:-1] if messages else []
                    logger.debug("connect#%s chat history %s", conn_id, lim_chat_history)
                    logger.debug("connect#%s user question %s", conn_id, question)
                    if lim_chat_history:
                        chat_history_support = ChatHistorySupport(question_handler, condense_template=chat_settings.langchain_condense_template)
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

                custom_docs = [Document(page_content=doc.text, metadata={"url": doc.metadata.url}) for doc in query_result.results]
                logger.info("connect#%s search query founded docs count=%s", conn_id, len(custom_docs))
                if chat_settings.doc_links_in_answer_enabled:
                    custom_docs = doc_links_answer_support.enrich_docs_with_source_links(custom_docs)

                # send source links
                if chat_settings.references_enabled:
                    source_links = [doc.metadata.url for doc in query_result.results]
                    docs_resp = ChatResponse(sender="bot", message=json.dumps(source_links), type="source_links")
                    await websocket.send_json(docs_resp.dict())

                langchain_template = chat_settings.langchain_template
                if chat_settings.doc_links_in_answer_enabled:
                    langchain_template = doc_links_answer_support.prompt_support(langchain_template)
                langchain_template = prompt_with_system_info(langchain_template)

                qa_chain = get_chain(clientVector, question_handler, stream_handler,
                                     '', langchain_template,
                                     custom_docs=custom_docs, temperature=chat_settings.open_ai_temperature, model_name=chat_settings.model_name,
                                     top_k_docs_for_context=chat_settings.langchain_chat_doc_count)

                result = await qa_chain.acall(
                   {"question": question, "chat_history": []}
                )
                chat_history.append((result['question'], result["answer"]))


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
async def lead_form_chat_endpoint_v2(websocket: WebSocket, form_id):
    await websocket.accept()
    conn_id = str(uuid.uuid4())
    try:
        logger.info("connect#%s start form#%s chatting", conn_id, form_id)
        owner = await aii_admin_api.get_user_by_leadform_id(form_id)
        logger.debug("connect#%s owner %s", conn_id, owner)
        api_key = general_openai_key
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

            # Call GPT-3.5 API
            openai.api_key = api_key

            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_input}],
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.4,
                stream=True
            )

            # Extract the text from the response
            openai_resp = ''
            start_resp = LeadFormChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            async for response_chunk in response:
                delta = response_chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    openai_resp += content
                    chat_resp = LeadFormChatResponse(
                        sender="bot",
                        message=content,
                        type="stream",
                    )
                    await websocket.send_json(chat_resp.dict())
            logger.debug('connect#%s openai response: "%s"', conn_id, openai_resp)
            end_resp = LeadFormChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())

            # increment user chat_messages_count per month
            incr_result = await aii_admin_api.increment_user_actions_count_per_month(owner['id'])
            logger.info(incr_result)

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
