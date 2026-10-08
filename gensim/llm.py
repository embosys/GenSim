"""OpenAI-compatible LLM client configured from the repository root .env."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from urllib.parse import urlparse


class LLMConfigurationError(ValueError):
    """Raised when the LLM environment is missing or invalid."""


class LLMResponseError(RuntimeError):
    """Raised when the provider returns incomplete or empty text."""


class LLMRequestError(RuntimeError):
    """Raised when an SDK request fails, without exposing request details."""


@dataclass(frozen=True)
class _Settings:
    model: str
    base_url: str | None
    timeout: float
    max_retries: int
    max_output_tokens: int
    temperature: float | None
    omit_temperature: bool
    extra_body: dict
    stream: bool
    supports_n: bool
    token_limit_param: str
    context_budget: int


_client = None
_settings: _Settings | None = None
_CORE_BODY_KEYS = {
    "model", "messages", "prompt", "max_tokens", "max_completion_tokens",
    "temperature", "n", "stream", "stream_options", "tools", "tool_choice",
    "response_format", "stop", "top_p", "frequency_penalty", "presence_penalty",
    "logit_bias", "seed", "user", "functions", "function_call", "api_key",
    "base_url", "timeout", "max_retries", "extra_body",
}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = os.environ.get(name)
    try:
        parsed = default if value is None else int(value)
    except ValueError:
        raise LLMConfigurationError(f"{name} must be an integer.") from None
    if parsed < minimum:
        raise LLMConfigurationError(f"{name} must be at least {minimum}.")
    return parsed


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    value = os.environ.get(name)
    try:
        parsed = default if value is None else float(value)
    except ValueError:
        raise LLMConfigurationError(f"{name} must be a number.") from None
    if not math.isfinite(parsed) or parsed < minimum:
        raise LLMConfigurationError(f"{name} must be a finite number at least {minimum}.")
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise LLMConfigurationError(f"{name} must be a boolean (true/false, yes/no, 1/0, or on/off).")


def _read_settings() -> tuple[_Settings, str]:
    model = os.environ.get("LLM_MODEL", "").strip()
    if not model:
        raise LLMConfigurationError("LLM_MODEL is required and must not be empty.")

    base_url = os.environ.get("LLM_BASE_URL", "").strip() or None
    if base_url:
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise LLMConfigurationError("LLM_BASE_URL must be an HTTP or HTTPS URL.")
    is_openai_endpoint = not base_url or urlparse(base_url).hostname == "api.openai.com"
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key and is_openai_endpoint:
        api_key = (
            os.environ.get("OPENAI_API_KEY", "").strip()
            or os.environ.get("OPENAI_KEY", "").strip()
        )
    if not api_key:
        raise LLMConfigurationError(
            "LLM_API_KEY is required (OPENAI_API_KEY or OPENAI_KEY may be used with api.openai.com)."
        )

    extra_body_value = os.environ.get("LLM_EXTRA_BODY", "").strip()
    try:
        extra_body = json.loads(extra_body_value) if extra_body_value else {}
    except json.JSONDecodeError:
        raise LLMConfigurationError("LLM_EXTRA_BODY must contain a valid JSON object.") from None
    if not isinstance(extra_body, dict):
        raise LLMConfigurationError("LLM_EXTRA_BODY must contain a JSON object.")
    reserved = sorted(key for key in extra_body if key.lower() in _CORE_BODY_KEYS)
    if reserved:
        raise LLMConfigurationError(
            "LLM_EXTRA_BODY cannot set core request fields: " + ", ".join(reserved)
        )

    temperature_value = os.environ.get("LLM_TEMPERATURE", "").strip()
    temperature = None
    omit_temperature = temperature_value.lower() in {"null", "none"}
    if temperature_value and not omit_temperature:
        try:
            temperature = float(temperature_value)
        except ValueError:
            raise LLMConfigurationError("LLM_TEMPERATURE must be a number when set.") from None
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise LLMConfigurationError("LLM_TEMPERATURE must be between 0 and 2.")

    token_limit_param = os.environ.get("LLM_TOKEN_LIMIT_PARAM", "max_tokens").strip()
    if token_limit_param not in {"max_tokens", "max_completion_tokens"}:
        raise LLMConfigurationError(
            "LLM_TOKEN_LIMIT_PARAM must be max_tokens or max_completion_tokens."
        )

    return (
        _Settings(
            model=model,
            base_url=base_url,
            timeout=_env_float("LLM_TIMEOUT", 120.0, minimum=0.001),
            max_retries=_env_int("LLM_MAX_RETRIES", 2, minimum=0),
            max_output_tokens=_env_int("LLM_MAX_OUTPUT_TOKENS", 2048),
            temperature=temperature,
            omit_temperature=omit_temperature,
            extra_body=extra_body,
            stream=_env_bool("LLM_STREAM", False),
            supports_n=_env_bool("LLM_SUPPORTS_N", False),
            token_limit_param=token_limit_param,
            context_budget=_env_int("LLM_CONTEXT_BUDGET", 6000),
        ),
        api_key,
    )


def configure_llm():
    """Load root .env settings and initialize the shared SDK client once."""
    global _client, _settings
    if _client is not None:
        return _client

    from dotenv import load_dotenv

    root_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(dotenv_path=root_env, override=False)
    settings, api_key = _read_settings()

    try:
        from openai import OpenAI
    except ImportError:
        raise LLMConfigurationError("The modern OpenAI Python SDK is required.") from None

    client_args = {
        "api_key": api_key,
        "timeout": settings.timeout,
        "max_retries": settings.max_retries,
    }
    if settings.base_url:
        client_args["base_url"] = settings.base_url
    try:
        _client = OpenAI(**client_args)
    except Exception as error:
        raise LLMConfigurationError(
            f"Could not initialize the OpenAI SDK client ({type(error).__name__})."
        ) from None
    _settings = settings
    return _client


def get_llm_model() -> str:
    configure_llm()
    return _settings.model


def get_context_budget() -> int:
    configure_llm()
    return _settings.context_budget


def _effective_temperature(temperature: float) -> float | None:
    if _settings.omit_temperature:
        return None
    value = _settings.temperature if _settings.temperature is not None else temperature
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("temperature must be a number.")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 2:
        raise ValueError("temperature must be between 0 and 2.")
    return value


def _output_tokens(max_tokens: int | None) -> int:
    value = _settings.max_output_tokens if max_tokens is None else max_tokens
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_tokens must be a positive integer or None.")
    return value


def _request_count(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")
    return n


def _request_args(max_tokens: int | None, temperature: float, n: int) -> dict:
    args = {
        "model": _settings.model,
        _settings.token_limit_param: _output_tokens(max_tokens),
        "stream": _settings.stream,
    }
    effective_temperature = _effective_temperature(temperature)
    if effective_temperature is not None:
        args["temperature"] = effective_temperature
    if _settings.supports_n:
        args["n"] = n
    if _settings.extra_body:
        args["extra_body"] = dict(_settings.extra_body)
    return args


def _stop_argument(stop):
    if stop is None:
        return None
    if isinstance(stop, str) and stop:
        return stop
    if isinstance(stop, (list, tuple)) and stop and all(
        isinstance(value, str) and value for value in stop
    ):
        return list(stop)
    raise ValueError("stop must be a non-empty string or list of non-empty strings.")


def _read_field(value, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _text_content(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part_text
            for part in value
            if isinstance((part_text := _read_field(part, "text")), str)
        )
    return ""


def _check_finish_reason(reason) -> None:
    if reason in {"length", "max_tokens", "incomplete"}:
        raise LLMResponseError(f"LLM response was truncated (finish_reason={reason}).")
    if reason != "stop":
        value = "missing" if reason is None else str(reason)
        raise LLMResponseError(
            f"LLM response did not finish normally (finish_reason={value})."
        )


def _validated_contents(contents: list[str], expected: int) -> list[str]:
    if len(contents) != expected:
        raise LLMResponseError(
            f"LLM returned {len(contents)} choices; expected {expected}."
        )
    for content in contents:
        if not content.strip():
            raise LLMResponseError("LLM returned empty final content.")
    return contents


def _raise_request_error(error: Exception) -> LLMRequestError:
    status_code = getattr(error, "status_code", None)
    status = f", HTTP {status_code}" if isinstance(status_code, int) else ""
    return LLMRequestError(
        f"LLM request failed ({type(error).__name__}{status}); SDK retries follow LLM_MAX_RETRIES."
    )


def _chat_contents(response, expected: int) -> list[str]:
    choices = _read_field(response, "choices", []) or []
    contents = []
    for choice in choices:
        reason = _read_field(choice, "finish_reason")
        _check_finish_reason(reason)
        message = _read_field(choice, "message", {})
        contents.append(_text_content(_read_field(message, "content")))
    return _validated_contents(contents, expected)


def _stream_chat_contents(stream, expected: int) -> list[str]:
    contents: dict[int, list[str]] = {}
    finish_reasons = {}
    for chunk in stream:
        for position, choice in enumerate(_read_field(chunk, "choices", []) or []):
            index = _read_field(choice, "index", position)
            contents.setdefault(index, [])
            delta = _read_field(choice, "delta", {})
            text = _text_content(_read_field(delta, "content"))
            if text:
                contents[index].append(text)
            reason = _read_field(choice, "finish_reason")
            if reason is not None:
                finish_reasons[index] = reason

    for index in contents:
        _check_finish_reason(finish_reasons.get(index))
    contents_in_order = ["".join(contents[index]) for index in sorted(contents)]
    return _validated_contents(contents_in_order, expected)


def _completion_contents(response, expected: int) -> list[str]:
    choices = _read_field(response, "choices", []) or []
    contents = []
    for choice in choices:
        reason = _read_field(choice, "finish_reason")
        _check_finish_reason(reason)
        contents.append(_text_content(_read_field(choice, "text")))
    return _validated_contents(contents, expected)


def _stream_completion_contents(stream, expected: int) -> list[str]:
    contents: dict[int, list[str]] = {}
    finish_reasons = {}
    for chunk in stream:
        for position, choice in enumerate(_read_field(chunk, "choices", []) or []):
            index = _read_field(choice, "index", position)
            contents.setdefault(index, [])
            text = _read_field(choice, "text")
            if isinstance(text, str):
                contents[index].append(text)
            reason = _read_field(choice, "finish_reason")
            if reason is not None:
                finish_reasons[index] = reason

    for index in contents:
        _check_finish_reason(finish_reasons.get(index))
    return _validated_contents(
        ["".join(contents[index]) for index in sorted(contents)], expected
    )


def _request_results(create, args: dict, n: int, *, chat: bool) -> list[str]:
    expected_per_call = n if _settings.supports_n else 1
    request_counts = [n] if _settings.supports_n else [1] * n
    results = []
    for request_n in request_counts:
        request_args = dict(args)
        if _settings.supports_n:
            request_args["n"] = request_n
        try:
            response = create(**request_args)
            if _settings.stream:
                contents = (
                    _stream_chat_contents(response, expected_per_call)
                    if chat
                    else _stream_completion_contents(response, expected_per_call)
                )
            else:
                contents = (
                    _chat_contents(response, expected_per_call)
                    if chat
                    else _completion_contents(response, expected_per_call)
                )
        except (LLMResponseError, LLMRequestError):
            raise
        except Exception as error:
            raise _raise_request_error(error) from None
        results.extend(contents)
    return results


def chat_completion(
    messages,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    n: int = 1,
    stop=None,
) -> list[str]:
    """Return final chat content for one or more completions."""
    stop = _stop_argument(stop)
    client = configure_llm()
    if isinstance(messages, (str, bytes)) or not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a non-empty list of chat message objects.")
    normalized_messages = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("each message must be a mapping with a string role.")
        normalized_messages.append(dict(message))
    n = _request_count(n)
    args = _request_args(max_tokens, temperature, n)
    if stop is not None:
        args["stop"] = stop
    return _request_results(
        lambda **kwargs: client.chat.completions.create(
            messages=normalized_messages, **kwargs
        ),
        args,
        n,
        chat=True,
    )


def completion(
    prompt: str,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    n: int = 1,
    stop=None,
) -> list[str]:
    """Return final text for a legacy text-completion endpoint."""
    stop = _stop_argument(stop)
    client = configure_llm()
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    n = _request_count(n)
    args = _request_args(max_tokens, temperature, n)
    args["prompt"] = prompt
    if stop is not None:
        args["stop"] = stop
    return _request_results(
        client.completions.create,
        args,
        n,
        chat=False,
    )
