import gc
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from types import SimpleNamespace
import weakref

from google import genai
import httpx
import pytest

from nanobanana_mcp_server.config.settings import AuthMethod, GeminiConfig, ServerConfig
from nanobanana_mcp_server.services import gemini_client as module
from nanobanana_mcp_server.core.exceptions import AuthenticationError
from nanobanana_mcp_server.services.gemini_client import GeminiClient


class OfflineRuntime:
    def __init__(self):
        self.transports = []
        self.clients = []
        self.requests = []
        self.constructor_calls = 0
        self.constructor_hook = None
        self.dispatch_hook = None
        self.lock = Lock()
        self.sdk = genai.Client

    def dispatch(self, request):
        with self.lock:
            self.requests.append(request)
        if self.dispatch_hook is not None:
            self.dispatch_hook()
        return httpx.Response(200, json={
            "candidates": [{"content": {
                "role": "model", "parts": [{"text": "offline fixture"}],
            }}],
        })

    def transport(self, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(self.dispatch), **kwargs)
        with self.lock:
            self.transports.append(client)
        return client

    def client(self, **kwargs):
        with self.lock:
            self.constructor_calls += 1
        if self.constructor_hook is not None:
            self.constructor_hook()
        client = self.sdk(**kwargs)
        with self.lock:
            if "httpx_client" not in kwargs["http_options"]:
                self.transports.append(client._api_client._httpx_client)
            self.clients.append(weakref.ref(client))
        return client

    def wrapper(self):
        return GeminiClient(
            ServerConfig(gemini_api_key="public-fixture-key", auth_method=AuthMethod.API_KEY),
            GeminiConfig(),
        )


@pytest.fixture
def runtime(monkeypatch):
    state = OfflineRuntime()
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda transport, request: state.dispatch(request))
    monkeypatch.setattr(module.genai, "Client", state.client)
    monkeypatch.setattr(module, "httpx", SimpleNamespace(Client=state.transport), raising=False)
    yield state
    for transport in state.transports:
        transport.close()


def test_replaces_a_closed_transport_before_the_next_generation(runtime):
    wrapper = runtime.wrapper()
    assert wrapper.generate_content(["public fixture"]).text == "offline fixture"
    runtime.transports[0].close()
    assert wrapper.generate_content(["public fixture"]).text == "offline fixture"
    assert runtime.constructor_calls == 2
    assert len(runtime.requests) == 2


def test_healthy_client_and_transport_are_reused(runtime):
    wrapper = runtime.wrapper()
    first = wrapper.client
    assert wrapper.generate_content(["public fixture"]).text == "offline fixture"
    assert wrapper.generate_content(["public fixture"]).text == "offline fixture"
    assert wrapper.client is first
    assert runtime.constructor_calls == 1
    assert len(runtime.transports) == 1
    assert runtime.requests[0].extensions["timeout"]["read"] == 60


def test_initialization_is_singleton_under_concurrent_calls(runtime):
    wrapper = runtime.wrapper()
    ready = Barrier(5)
    entered = Event()
    release = Event()
    all_constructors = Event()

    def block_constructor():
        entered.set()
        if runtime.constructor_calls == 4:
            all_constructors.set()
        assert release.wait(3)

    runtime.constructor_hook = block_constructor

    def access():
        ready.wait(timeout=3)
        return wrapper.client

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(access) for _ in range(4)]
        ready.wait(timeout=3)
        assert entered.wait(3)
        all_constructors.wait(0.2)
        release.set()
        clients = [future.result(timeout=3) for future in futures]
    assert runtime.constructor_calls == 1
    assert all(client is clients[0] for client in clients)


def test_sdk_client_is_kept_alive_through_the_request(runtime):
    wrapper = runtime.wrapper()

    def during_dispatch():
        wrapper._client = None
        gc.collect()
        assert runtime.clients[0]() is not None

    runtime.dispatch_hook = during_dispatch
    assert wrapper.generate_content(["public fixture"]).text == "offline fixture"


def test_a_dispatch_failure_is_not_automatically_replayed(runtime):
    wrapper = runtime.wrapper()

    def fail_dispatch():
        raise RuntimeError("public fixture: dispatch outcome unknown")

    runtime.dispatch_hook = fail_dispatch
    with pytest.raises(RuntimeError, match="dispatch outcome unknown"):
        wrapper.generate_content(["public fixture"])
    assert runtime.constructor_calls == 1
    assert len(runtime.requests) == 1


def test_missing_auth_does_not_allocate_a_transport(runtime):
    wrapper = runtime.wrapper()
    wrapper.config.gemini_api_key = None
    with pytest.raises(AuthenticationError, match="API key is required"):
        _ = wrapper.client
    assert runtime.transports == []
    assert runtime.constructor_calls == 0


def test_failed_sdk_initialization_closes_the_new_owned_transport(runtime, monkeypatch):
    wrapper = runtime.wrapper()

    def reject_client(**kwargs):
        raise ValueError("public fixture: initialization failed")

    monkeypatch.setattr(module.genai, "Client", reject_client)
    with pytest.raises(ValueError, match="initialization failed"):
        _ = wrapper.client
    assert len(runtime.transports) == 1
    assert runtime.transports[0].is_closed
    assert wrapper._client is None


def test_sdk_close_does_not_close_the_wrapper_owned_transport(runtime):
    wrapper = runtime.wrapper()
    wrapper.client.close()
    assert wrapper.generate_content(["public fixture"]).text == "offline fixture"
    assert runtime.constructor_calls == 1
    assert not runtime.transports[0].is_closed


def test_owned_transport_closes_when_its_wrapper_is_collected(runtime):
    wrapper = runtime.wrapper()
    _ = wrapper.client
    transport = runtime.transports[0]
    del wrapper
    gc.collect()
    assert transport.is_closed
