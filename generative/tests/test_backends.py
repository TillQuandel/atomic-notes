"""Tests für _subscription_backend und _litellm_backend."""
import asyncio
import json
import subprocess
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from agents._subscription_backend import call_full as sub_call_full
from agents._subscription_backend import call_full_async as sub_call_full_async
from agents._subscription_backend import _to_cli_model
import agents._subscription_backend as sub_backend
from agents.base import CallResult
from runtime_config import load_runtime_config


@pytest.fixture(autouse=True)
def clear_base_runtime_config():
    import agents.base as base_mod
    base_mod.clear_llm_runtime_config()
    yield
    base_mod.clear_llm_runtime_config()


# --- _to_cli_model Mapping ---

def test_to_cli_model_maps_opus():
    assert _to_cli_model("anthropic/claude-opus-4-7") == "opus"

def test_to_cli_model_maps_haiku():
    assert _to_cli_model("anthropic/claude-haiku-4-5-20251001") == "haiku"

def test_to_cli_model_maps_sonnet():
    assert _to_cli_model("anthropic/claude-sonnet-4-6") == "sonnet"

def test_to_cli_model_passthrough_unknown():
    assert _to_cli_model("ollama/llama3") == "ollama/llama3"

def test_to_cli_model_passthrough_openai():
    assert _to_cli_model("openai/gpt-4o") == "openai/gpt-4o"


# --- Subscription sync call ---

def test_sub_call_full_returns_callresult(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOMIC_AGENT_BACKEND", "subscription")

    fake_response = json.dumps({
        "result": "hello world",
        "is_error": False,
        "duration_ms": 123,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    })
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = fake_response
    mock_proc.stderr = ""

    with patch("subprocess.run", return_value=mock_proc):
        result = sub_call_full("test prompt", model="anthropic/claude-opus-4-7", agent="test")

    assert isinstance(result, CallResult)
    assert result.text == "hello world"
    assert result.input_tokens == 10
    assert result.output_tokens == 5


def test_sub_call_full_does_not_retry_timeout_by_default(monkeypatch):
    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["claude"], timeout=300),
    ) as run_mock:
        with pytest.raises(RuntimeError, match="Timeout nach"):
            sub_call_full("test prompt", model="anthropic/claude-opus-4-7", agent="test")

    assert run_mock.call_count == 1


def test_sub_call_full_retries_timeout_when_enabled(monkeypatch):
    fake_response = json.dumps({
        "result": "recovered",
        "is_error": False,
        "duration_ms": 123,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    })
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = fake_response
    mock_proc.stderr = ""

    monkeypatch.setattr(sub_backend, "_TIMEOUT_RETRIES", 1)

    with patch(
        "subprocess.run",
        side_effect=[
            subprocess.TimeoutExpired(cmd=["claude"], timeout=300),
            mock_proc,
        ],
    ) as run_mock, patch("time.sleep"):
        result = sub_call_full("test prompt", model="anthropic/claude-opus-4-7", agent="test")

    assert result.text == "recovered"
    assert run_mock.call_count == 2


def test_sub_call_full_uses_runtime_timeout_args():
    fake_response = json.dumps({
        "result": "custom timeout",
        "is_error": False,
        "duration_ms": 123,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    })
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = fake_response
    mock_proc.stderr = ""

    with patch("subprocess.run", return_value=mock_proc) as run_mock:
        result = sub_call_full(
            "test prompt",
            model="anthropic/claude-opus-4-7",
            agent="test",
            call_timeout_sec=123,
            timeout_retries=0,
        )

    assert result.text == "custom timeout"
    assert run_mock.call_args.kwargs["timeout"] == 123


def test_sub_call_full_runtime_timeout_retries_override_module_default(monkeypatch):
    fake_response = json.dumps({
        "result": "runtime recovered",
        "is_error": False,
        "duration_ms": 123,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    })
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = fake_response
    mock_proc.stderr = ""
    monkeypatch.setattr(sub_backend, "_TIMEOUT_RETRIES", 0)

    with patch(
        "subprocess.run",
        side_effect=[
            subprocess.TimeoutExpired(cmd=["claude"], timeout=111),
            mock_proc,
        ],
    ) as run_mock, patch("time.sleep"):
        result = sub_call_full(
            "test prompt",
            model="anthropic/claude-opus-4-7",
            agent="test",
            call_timeout_sec=111,
            timeout_retries=1,
        )

    assert result.text == "runtime recovered"
    assert run_mock.call_count == 2


# --- Subscription async call ---

@pytest.mark.asyncio
async def test_sub_call_full_async_returns_callresult(monkeypatch):
    monkeypatch.setenv("ATOMIC_AGENT_BACKEND", "subscription")

    fake_response = json.dumps({
        "result": "async hello",
        "is_error": False,
        "duration_ms": 99,
        "usage": {"input_tokens": 8, "output_tokens": 3,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    })

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(
        return_value=(fake_response.encode(), b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await sub_call_full_async("test prompt", model="anthropic/claude-opus-4-7", agent="test")

    assert result.text == "async hello"
    assert result.input_tokens == 8


@pytest.mark.asyncio
async def test_sub_call_full_async_does_not_retry_timeout_by_default(monkeypatch):
    timed_out_proc = AsyncMock()
    timed_out_proc.returncode = None
    timed_out_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    timed_out_proc.kill = MagicMock()
    timed_out_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=timed_out_proc) as create_mock:
        with pytest.raises(RuntimeError, match="Timeout nach"):
            await sub_call_full_async(
                "test prompt",
                model="anthropic/claude-opus-4-7",
                agent="test",
            )

    assert create_mock.call_count == 1
    timed_out_proc.kill.assert_called_once()
    timed_out_proc.wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_sub_call_full_async_retries_timeout_when_enabled(monkeypatch):
    fake_response = json.dumps({
        "result": "async recovered",
        "is_error": False,
        "duration_ms": 99,
        "usage": {"input_tokens": 8, "output_tokens": 3,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    })

    timed_out_proc = AsyncMock()
    timed_out_proc.returncode = None
    timed_out_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    timed_out_proc.kill = MagicMock()
    timed_out_proc.wait = AsyncMock()

    recovered_proc = AsyncMock()
    recovered_proc.returncode = 0
    recovered_proc.communicate = AsyncMock(
        return_value=(fake_response.encode(), b"")
    )

    monkeypatch.setattr(sub_backend, "_TIMEOUT_RETRIES", 1)

    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=[timed_out_proc, recovered_proc],
    ) as create_mock, patch("asyncio.sleep", new_callable=AsyncMock):
        result = await sub_call_full_async(
            "test prompt",
            model="anthropic/claude-opus-4-7",
            agent="test",
        )

    assert result.text == "async recovered"
    assert create_mock.call_count == 2
    timed_out_proc.kill.assert_called_once()
    timed_out_proc.wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_sub_call_full_async_uses_runtime_timeout_args(monkeypatch):
    fake_response = json.dumps({
        "result": "async custom timeout",
        "is_error": False,
        "duration_ms": 99,
        "usage": {"input_tokens": 8, "output_tokens": 3,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    })

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(
        return_value=(fake_response.encode(), b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
            patch("asyncio.wait_for", wraps=asyncio.wait_for) as wait_mock:
        result = await sub_call_full_async(
            "test prompt",
            model="anthropic/claude-opus-4-7",
            agent="test",
            call_timeout_sec=222,
            timeout_retries=0,
        )

    assert result.text == "async custom timeout"
    assert wait_mock.call_args.kwargs["timeout"] == 222


# --- litellm-Backend ---

from agents._litellm_backend import call_full as lit_call_full
from agents._litellm_backend import call_full_async as lit_call_full_async
import agents._litellm_backend as lit_backend


def _make_litellm_response(text: str, in_tok: int = 10, out_tok: int = 5,
                            cache_read: int = 0, cache_create: int = 0):
    usage = MagicMock()
    usage.prompt_tokens = in_tok
    usage.completion_tokens = out_tok
    usage.cache_read_input_tokens = cache_read
    usage.cache_creation_input_tokens = cache_create

    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def test_lit_call_full_returns_callresult():
    fake_resp = _make_litellm_response("litellm response", in_tok=12, out_tok=7)

    with patch("litellm.completion", return_value=fake_resp):
        result = lit_call_full("test prompt", model="anthropic/claude-opus-4-7", agent="test")

    assert isinstance(result, CallResult)
    assert result.text == "litellm response"
    assert result.input_tokens == 12
    assert result.output_tokens == 7


def test_lit_call_full_passes_model_unchanged():
    """litellm-Backend darf das Modell NICHT auf CLI-Shorthand mappen."""
    fake_resp = _make_litellm_response("ok")
    captured = {}

    def fake_completion(model, messages, **kwargs):
        captured["model"] = model
        return fake_resp

    with patch("litellm.completion", side_effect=fake_completion):
        lit_call_full("prompt", model="openai/gpt-4o", agent="test")

    assert captured["model"] == "openai/gpt-4o"


def test_lit_call_full_passes_gemini_model():
    fake_resp = _make_litellm_response("ok")
    captured = {}

    def fake_completion(model, messages, **kwargs):
        captured["model"] = model
        return fake_resp

    with patch("litellm.completion", side_effect=fake_completion):
        lit_call_full("prompt", model="gemini/gemini-2.0-flash", agent="test")

    assert captured["model"] == "gemini/gemini-2.0-flash"


def test_lit_call_full_cache_tokens():
    fake_resp = _make_litellm_response("cached", cache_read=500, cache_create=100)

    with patch("litellm.completion", return_value=fake_resp):
        result = lit_call_full("prompt", model="anthropic/claude-opus-4-7", agent="test")

    assert result.cache_read_tokens == 500
    assert result.cache_creation_tokens == 100


def test_lit_call_full_uses_runtime_timeout_arg():
    fake_resp = _make_litellm_response("ok")
    captured = {}

    def fake_completion(model, messages, **kwargs):
        captured.update(kwargs)
        return fake_resp

    with patch("litellm.completion", side_effect=fake_completion):
        result = lit_call_full(
            "prompt",
            model="anthropic/claude-opus-4-7",
            agent="test",
            call_timeout_sec=77,
            timeout_retries=1,
        )

    assert result.text == "ok"
    assert captured["request_timeout"] == 77
    assert captured["num_retries"] == lit_backend._MAX_RETRIES


@pytest.mark.asyncio
async def test_lit_call_full_async_uses_runtime_timeout_arg():
    fake_resp = _make_litellm_response("async ok")
    captured = {}

    async def fake_acompletion(model, messages, **kwargs):
        captured.update(kwargs)
        return fake_resp

    with patch("litellm.acompletion", side_effect=fake_acompletion):
        result = await lit_call_full_async(
            "prompt",
            model="anthropic/claude-opus-4-7",
            agent="test",
            call_timeout_sec=88,
            timeout_retries=1,
        )

    assert result.text == "async ok"
    assert captured["request_timeout"] == 88
    assert captured["num_retries"] == lit_backend._MAX_RETRIES


@pytest.mark.asyncio
async def test_lit_call_full_async_returns_callresult():
    fake_resp = _make_litellm_response("async lit", in_tok=9, out_tok=4)

    async def fake_acompletion(model, messages, **kwargs):
        return fake_resp

    with patch("litellm.acompletion", side_effect=fake_acompletion):
        result = await lit_call_full_async("async prompt", model="anthropic/claude-haiku-4-5-20251001", agent="test")

    assert result.text == "async lit"
    assert result.input_tokens == 9


# --- Dispatch in base.py ---

def test_dispatch_subscription_calls_subprocess(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOMIC_AGENT_BACKEND", "subscription")
    # config und base neu laden damit BACKEND-Env-Wert greift
    import importlib
    import config as cfg_mod
    importlib.reload(cfg_mod)
    import agents._subscription_backend as sub_mod
    importlib.reload(sub_mod)
    import agents.base as base_mod
    importlib.reload(base_mod)

    (tmp_path / ".cache" / "llm").mkdir(parents=True)
    (tmp_path / ".cache" / "runs").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    fake_response = json.dumps({
        "result": "sub dispatch", "is_error": False, "duration_ms": 50,
        "usage": {"input_tokens": 1, "output_tokens": 1,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    })
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = fake_response
    mock_proc.stderr = ""

    with patch("subprocess.run", return_value=mock_proc):
        result = base_mod.call_claude_full("hello", model="anthropic/claude-opus-4-7",
                                           agent="test", use_cache=False)

    assert result.text == "sub dispatch"


def test_dispatch_subscription_uses_runtime_config(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOMIC_AGENT_BACKEND", "subscription")
    import importlib
    import config as cfg_mod
    importlib.reload(cfg_mod)
    import agents._subscription_backend as sub_mod
    importlib.reload(sub_mod)
    import agents.base as base_mod
    importlib.reload(base_mod)

    (tmp_path / ".cache" / "llm").mkdir(parents=True)
    (tmp_path / ".cache" / "runs").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    fake_response = json.dumps({
        "result": "runtime dispatch", "is_error": False, "duration_ms": 50,
        "usage": {"input_tokens": 1, "output_tokens": 1,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    })
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = fake_response
    mock_proc.stderr = ""

    runtime_config = load_runtime_config(env={
        "ATOMIC_AGENT_CALL_TIMEOUT": "123",
        "ATOMIC_AGENT_TIMEOUT_RETRIES": "1",
    })
    base_mod.set_llm_runtime_config(runtime_config)

    with patch("subprocess.run", return_value=mock_proc) as run_mock:
        result = base_mod.call_claude_full(
            "hello",
            model="anthropic/claude-opus-4-7",
            agent="test",
            use_cache=False,
        )

    assert result.text == "runtime dispatch"
    assert run_mock.call_args.kwargs["timeout"] == 123


def test_base_runtime_config_clear_restores_backend_defaults():
    import agents.base as base_mod

    runtime_config = load_runtime_config(env={
        "ATOMIC_AGENT_CALL_TIMEOUT": "123",
        "ATOMIC_AGENT_TIMEOUT_RETRIES": "1",
    })
    base_mod.set_llm_runtime_config(runtime_config)
    assert base_mod._backend_runtime_kwargs() == {
        "call_timeout_sec": 123,
        "timeout_retries": 1,
    }

    base_mod.clear_llm_runtime_config()

    assert base_mod._backend_runtime_kwargs() == {}


def test_dispatch_litellm_calls_litellm(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOMIC_AGENT_BACKEND", "litellm")
    import importlib
    import config as cfg_mod
    importlib.reload(cfg_mod)
    import agents._litellm_backend as lit_mod
    importlib.reload(lit_mod)
    import agents.base as base_mod
    importlib.reload(base_mod)

    (tmp_path / ".cache" / "llm").mkdir(parents=True)
    (tmp_path / ".cache" / "runs").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    fake_resp = _make_litellm_response("lit dispatch", in_tok=3, out_tok=2)

    with patch("litellm.completion", return_value=fake_resp):
        result = base_mod.call_claude_full("hello", model="anthropic/claude-opus-4-7",
                                           agent="test", use_cache=False)

    assert result.text == "lit dispatch"
