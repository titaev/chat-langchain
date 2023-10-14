import asyncio
import time

import aiohttp
import pytest
import pytest_asyncio


@pytest_asyncio.fixture(scope='session')
async def http_session():
    """
    Asynchronously create and yield an aiohttp client session for the test session.

    Yields:
        session: The aiohttp client session.
    """
    async with aiohttp.ClientSession() as session:
        yield session