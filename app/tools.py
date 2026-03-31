from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict
from uuid import uuid4


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Dict[str, Any]]] = {}

    def register(self, name: str, fn: Callable[..., Dict[str, Any]]) -> None:
        self._tools[name] = fn

    def call(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        if name not in self._tools:
            raise ValueError(f"tool not found: {name}")
        return self._tools[name](**kwargs)


def create_ticket(
    subject: str,
    sender: str,
    description: str,
    priority: str = "medium",
) -> Dict[str, Any]:
    ticket_id = f"TIC-{uuid4().hex[:8].upper()}"
    return {
        "ticket_id": ticket_id,
        "status": "created",
        "priority": priority,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "subject": subject,
        "sender": sender,
        "description_preview": description[:80],
    }


def start_approval_flow(
    sender: str,
    title: str,
    details: str,
    amount: float | None = None,
) -> Dict[str, Any]:
    flow_id = f"APR-{uuid4().hex[:8].upper()}"
    return {
        "flow_id": flow_id,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "requester": sender,
        "title": title,
        "amount": amount,
        "details_preview": details[:80],
    }


def send_auto_reply(recipient: str, subject: str, body: str) -> Dict[str, Any]:
    return {
        "message_id": f"MSG-{uuid4().hex[:8].upper()}",
        "status": "queued",
        "recipient": recipient,
        "subject": subject,
        "body_preview": body[:120],
    }
