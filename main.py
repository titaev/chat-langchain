"""Main entrypoint for the app."""
import logging
import pickle
import json
import httpx
from operator import itemgetter
from pathlib import Path
from typing import Optional
import openai

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import PlainTextResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.staticfiles import StaticFiles
from langchain.vectorstores import VectorStore
from websockets.exceptions import ConnectionClosedError

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse, LeadFormChatResponse
from usersData import getUsersData
from aii_admin_service import AiiAdminApi

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None
httpx_session: httpx.AsyncClient
aii_admin_api: AiiAdminApi

# app.add_middleware(HTTPSRedirectMiddleware)
# app.mount("/.well-known/pki-validation", StaticFiles(directory="./.well-known/pki-validation"), name="static")


@app.on_event("startup")
async def startup():
    global httpx_session
    httpx_session = httpx.AsyncClient()
    global aii_admin_api
    aii_admin_api = AiiAdminApi(httpx_session)


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


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    users_data = getUsersData()

    while True:
        try:
            # Receive and send back the client message
            request = await websocket.receive_text()
            question, clientId, persistHistory = itemgetter('question', 'clientId', 'persistHistory')(json.loads(request))
            user = users_data.get(clientId)
            if not(user):
                resp = ChatResponse(
                    sender="bot",
                    message="Client with such id doesn't exist.",
                    type="error",
                )
                await websocket.send_json(resp.dict())
                continue
            clientVector = VectorStore()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            lim_chat_history = chat_history[-5:] if persistHistory else []

            qa_chain = get_chain(clientVector, question_handler, stream_handler, user['condense_template'], user['template'], custom_results="Stability Balls (Swiss Balls) • Balance products; Bosu Ball, Wobble boards and balance boards • Dumbbells • Roman Chair A stability ball is not only inexpensive, easy to use and readily available but it also improves balance, coordination and strengthens those hard to get to muscles. To ensure that the correct size stability ball is used, sit on top of the ball with the feet hip width apart. The knees should be level with the hips, and a 90-degree angle should be formed at the knee joint between the legs and thighs. Based on height, the following stability ball sizes are recommended: • 4’ 11”- 5’ 3” you should be using a 55cm ball • 5’ 4”-5’ 10” you should be using a 65cm ball • 5’11” and up you should be using a 75 cm ball. 99 Buy Now Costco Pickleball Paddle Set (Purchase Online Or In Person) Bundle Includes- 2 Latitude Paddles, 3 Balls, and 1 Bag.... Shop • Paddles • Gear • Brands • Programs • Watch • Resources • Help • Blog Information • Contact us • Start a warranty claim • Refund Policy • Shipping Policy • FAQ • Terms of Service • Privacy Policy • Trademarks • Sitemap All paddles and equipment are only intended to be used in Pickleball play with a Pickleball ball. They are not intended to be used as toys or by infants and children. Close Customer Login If you are already registered, please log in. Shipping Taxes and shipping fee will be calculated at checkout  Source page: https://www.selkirk. add as I did a ball mesh bag hold a few extra balls or other items so I was not digging around for them. 3) include a fence hook 4) color the inside of the bag in a bright color to easily see/find items in bag. This may not be as bad with the blue bag. It works, and there was no misleading of info in the description and I do like it, could be better than a normal backpack though. Shop • Paddles • Gear • Brands • Programs • Watch • Resources • Help • Blog Information • Contact us • Start a warranty claim • Refund Policy • Shipping Policy • FAQ • Terms of Service • Privacy Policy • Trademarks • Sitemap")

            result = await qa_chain.acall(
               {"question": question, "chat_history": lim_chat_history}
            )
            chat_history.append((result['question'], result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


@app.websocket("/chat/lead_form/{form_id}")
async def websocket_endpoint(websocket: WebSocket, form_id):
    await websocket.accept()
    try:
        api_key = await aii_admin_api.get_openai_key_by_leadform_id(form_id)
        if not api_key:
            raise ConnectionClosedError

        while True:
            data = await websocket.receive_text()
            user_input = json.loads(data)["text"]

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
            async for response_chunk in response:
                delta = response_chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    await websocket.send_text(content)

            await websocket.send_text(json.dumps({'type': 'response_end'}))

    except ConnectionClosedError:
        await websocket.close()


@app.websocket("/chat/v2/lead_form/{form_id}")
async def lead_form_chat_endpoint_v2(websocket: WebSocket, form_id):
    await websocket.accept()
    try:
        api_key = await aii_admin_api.get_openai_key_by_leadform_id(form_id)
        if not api_key:
            resp = LeadFormChatResponse(
                sender="bot",
                message="User with this form doesn't have api_key in aii_admin",
                type="error",
            )
            await websocket.send_json(resp.dict())
            raise ConnectionClosedError

        while True:
            data = await websocket.receive_text()
            user_input = json.loads(data)["text"]

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
            start_resp = LeadFormChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            async for response_chunk in response:
                delta = response_chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    chat_resp = LeadFormChatResponse(
                        sender="bot",
                        message=content,
                        type="stream",
                    )
                    await websocket.send_json(chat_resp.dict())

            end_resp = LeadFormChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())

    except ConnectionClosedError:
        await websocket.close()
    except openai.OpenAIError as error:
        end_resp = LeadFormChatResponse(sender="bot", message=str(error), type="error")
        await websocket.send_json(end_resp.dict())
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
