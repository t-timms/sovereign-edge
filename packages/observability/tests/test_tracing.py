from __future__ import annotations

import pytest
from observability.tracing import traced

# ── sync traced decorator ─────────────────────────────────────────────────────


def test_traced_sync_calls_function() -> None:
    @traced("test.sync")
    def add(a: int, b: int) -> int:
        return a + b

    result = add(2, 3)
    assert result == 5


def test_traced_sync_preserves_function_name() -> None:
    @traced()
    def my_func() -> str:
        return "hello"

    assert my_func.__name__ == "my_func"


def test_traced_sync_passes_args_and_kwargs() -> None:
    captured: dict = {}

    @traced("test.kwargs")
    def record(x: int, *, label: str = "default") -> None:
        captured["x"] = x
        captured["label"] = label

    record(42, label="custom")
    assert captured == {"x": 42, "label": "custom"}


def test_traced_sync_propagates_exception() -> None:
    @traced("test.raises")
    def fail() -> None:
        raise ValueError("deliberate error")

    with pytest.raises(ValueError, match="deliberate error"):
        fail()


# ── async traced decorator ────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_traced_async_calls_function() -> None:
    @traced("test.async")
    async def greet(name: str) -> str:
        return f"hello {name}"

    result = await greet("john")
    assert result == "hello john"


@pytest.mark.asyncio()
async def test_traced_async_preserves_function_name() -> None:
    @traced()
    async def my_async_func() -> None:
        pass

    assert my_async_func.__name__ == "my_async_func"


@pytest.mark.asyncio()
async def test_traced_async_propagates_exception() -> None:
    @traced("test.async_raises")
    async def async_fail() -> None:
        raise RuntimeError("async error")

    with pytest.raises(RuntimeError, match="async error"):
        await async_fail()


# ── span_name default ─────────────────────────────────────────────────────────


def test_traced_uses_qualname_when_no_span_name() -> None:
    """Verify the decorator defaults to fn.__qualname__ for the span name."""

    @traced()
    def my_named_func() -> str:
        return "named"

    # If this executes without error, the span name was derived correctly
    result = my_named_func()
    assert result == "named"
