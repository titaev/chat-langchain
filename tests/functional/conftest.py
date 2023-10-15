import websockets
import aiohttp
import pytest
import asyncio
import pytest_asyncio
from .settings import test_settings


@pytest.fixture(scope="session")
def event_loop():
    """
    Create and yield a new event loop for the test session.

    :yield: aiohttp client session.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope='session')
async def lead_form_ws_session():
    """
    Asynchronously create and yield a ws client session for the test session for lead_form.

    Yields:
        session: The aiohttp ws session.
    """
    async with websockets.connect(f"{test_settings.service_url}/chat/v2/lead_form/feb071e0-b46e-4015-89a8-60cefaa000e4") as websocket:
        yield websocket
