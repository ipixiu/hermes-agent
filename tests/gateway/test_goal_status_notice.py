from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


class FakeAdapter:
    def __init__(self):
        self.calls = []
        self.callbacks = {}
        self._active_sessions = {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SimpleNamespace(success=True)

    def register_post_delivery_callback(self, session_key, callback, *, generation=None):
        self.callbacks[session_key] = (generation, callback)


def _goal_continuation_event(source, goal="finish the task"):
    return MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal=goal),
        message_type=MessageType.TEXT,
        source=source,
    )


@pytest.mark.asyncio
async def test_goal_status_notice_uses_adapter_send_with_thread_metadata():
    """Regression: /goal judge status must use BasePlatformAdapter.send().

    The old implementation checked for a non-existent send_message() method,
    so the goal could be marked done in state_meta without the visible
    "✓ Goal achieved" status line being delivered to Discord/Telegram.
    """
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = FakeAdapter()
    runner.adapters = {Platform.DISCORD: adapter}

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="parent-channel",
        thread_id="thread-123",
    )

    await runner._send_goal_status_notice(source, "✓ Goal achieved: done")

    assert adapter.calls == [
        {
            "chat_id": "parent-channel",
            "content": "✓ Goal achieved: done",
            "reply_to": None,
            "metadata": {"thread_id": "thread-123"},
        }
    ]


@pytest.mark.asyncio
async def test_goal_status_notice_defers_until_post_delivery_callback():
    """Regression: goal status must appear after the agent's visible reply.

    _post_turn_goal_continuation runs before BasePlatformAdapter sends the
    returned final response. It should therefore register a post-delivery
    callback, not send the judge status immediately.
    """
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = FakeAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner.config = SimpleNamespace(group_sessions_per_user=True, thread_sessions_per_user=False)

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="parent-channel",
        thread_id="thread-123",
        user_id="user-1",
    )

    await runner._defer_goal_status_notice_after_delivery(source, "✓ Goal achieved: done")

    assert adapter.calls == []
    assert len(adapter.callbacks) == 1

    _, callback = next(iter(adapter.callbacks.values()))
    result = callback()
    if hasattr(result, "__await__"):
        await result

    assert adapter.calls == [
        {
            "chat_id": "parent-channel",
            "content": "✓ Goal achieved: done",
            "reply_to": None,
            "metadata": {"thread_id": "thread-123"},
        }
    ]


def test_clear_goal_pending_continuations_removes_slot_and_overflow_only():
    """Regression: /goal pause/clear must cancel queued self-continuations.

    A user-issued /goal pause can arrive after the judge queued the next
    continuation but before that queued turn runs.  The queued synthetic goal
    continuation should be removed without dropping normal user /queue items.
    """
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = FakeAdapter()
    adapter._pending_messages = {}
    runner._queued_events = {}

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="parent-channel",
        thread_id="thread-123",
    )
    session_key = "discord:parent-channel:thread-123"
    normal_event = MessageEvent(
        text="normal queued user message",
        message_type=MessageType.TEXT,
        source=source,
    )

    adapter._pending_messages[session_key] = _goal_continuation_event(source)
    runner._queued_events[session_key] = [
        normal_event,
        _goal_continuation_event(source, goal="second continuation"),
    ]

    removed = runner._clear_goal_pending_continuations(session_key, adapter)

    assert removed == 2
    assert adapter._pending_messages.get(session_key) is None
    assert runner._queued_events[session_key] == [normal_event]


@pytest.mark.asyncio
async def test_goal_resume_returns_status_line_after_pause(hermes_home):
    """/goal resume should surface the refreshed active status line."""
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = SimpleNamespace(goals={"max_turns": 5})
    runner.session_store = SimpleNamespace()
    runner.session_store.get_or_create_session = lambda source: SimpleNamespace(
        session_id="goal-resume-sid"
    )

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="parent-channel",
        thread_id="thread-123",
    )
    event = MessageEvent(
        text="/goal resume",
        message_type=MessageType.TEXT,
        source=source,
    )

    from hermes_cli.goals import GoalManager

    mgr = GoalManager("goal-resume-sid", default_max_turns=5)
    mgr.set("finish the task")
    mgr.pause(reason="user-paused")

    result = await runner._handle_goal_command(event)

    assert "Goal resumed" in result
    assert "Goal (active" in result
    assert "0/5 turns" in result
    assert result.index("Goal resumed") < result.index("Goal (active")
    assert result.index("Goal (active") < result.index("Send any message")
