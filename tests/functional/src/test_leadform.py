import pytest
import json


@pytest.mark.asyncio
async def test_lead_form_alive(lead_form_ws_session):
    await lead_form_ws_session.send(json.dumps({"text": "hello"}))

    received_messages = []
    while True:
        msg = await lead_form_ws_session.recv()
        response = json.loads(msg)
        received_messages.append(response)
        if response['type'] == "end":
            break

    assert received_messages[0]['type'] == "start"
    assert received_messages[1]['type'] == "stream"
    assert received_messages[-1]['type'] == "end"






