from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dataclasses import dataclass, field

TERMINAL_STOP_REASONS = {"end_turn", "max_tokens", "stop_sequence", "refusal"}


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value

@dataclass
class RetrievalContext:
    doc_id: str
    text: str
    score: float
    collection: str
    query: str
    turn_index: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Step:
    """One assistant response within the agent loop, plus whatever
    was sent back to resolve it (if it wasn't terminal)."""
    assistant_content: list[Any]        # raw response.content blocks, stored verbatim
    stop_reason: str
    tool_results: list[dict] | None = None  # tool_result blocks, if this step needed a follow-up


@dataclass
class Turn:
    """Everything from one piece of user input through however many
    agent-loop iterations it takes to reach a terminal stop_reason."""
    user_input: str
    steps: list[Step] = field(default_factory=list)

    def add_step(self, assistant_content: list[Any], stop_reason: str,
                 tool_results: list[dict] | None = None) -> None:
        self.steps.append(Step(assistant_content, stop_reason, tool_results))

    @property
    def is_complete(self) -> bool:
        return bool(self.steps) and self.steps[-1].stop_reason in TERMINAL_STOP_REASONS

    @property
    def final_text(self) -> str:
        """Text from the last step — what you'd actually show the user."""
        last = self.steps[-1]
        return "".join(
            block.get("text", "") if isinstance(block, dict) else block.text
            for block in last.assistant_content
            if (
                block.get("type") if isinstance(block, dict)
                else getattr(block, "type", None)
            ) == "text"
        )


@dataclass
class Context:
    turns: list[Turn] = field(default_factory=list)
    retrieval_contexts: list[RetrievalContext] = field(default_factory=list)

    def append_turn(self, turn: Turn | None = None, user_input: str | None = None) -> Turn:
        if turn is None:
            turn = Turn(user_input=user_input or "")
        self.turns.append(turn)
        return turn

    def append_retrieval_context(self, doc_id: str, text: str, score: float, collection: str, query: str, turn_index: int) -> None:
        self.retrieval_contexts.append(
            RetrievalContext(
                doc_id=doc_id,
                text=text,
                score=score,
                collection=collection,
                query=query,
                turn_index=turn_index,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the full agent context."""
        return {
            "turns": [
                {
                    "user_input": turn.user_input,
                    "steps": [
                        {
                            "assistant_content": _json_safe(step.assistant_content),
                            "stop_reason": step.stop_reason,
                            "tool_results": _json_safe(step.tool_results),
                        }
                        for step in turn.steps
                    ],
                }
                for turn in self.turns
            ],
            "retrieval_contexts": [
                {
                    "doc_id": item.doc_id,
                    "text": item.text,
                    "score": item.score,
                    "collection": item.collection,
                    "query": item.query,
                    "turn_index": item.turn_index,
                    "timestamp": item.timestamp.isoformat(),
                }
                for item in self.retrieval_contexts
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Context:
        """Rebuild a context from its JSON-compatible representation."""
        turns = [
            Turn(
                user_input=turn_data["user_input"],
                steps=[
                    Step(
                        assistant_content=step_data.get("assistant_content", []),
                        stop_reason=step_data["stop_reason"],
                        tool_results=step_data.get("tool_results"),
                    )
                    for step_data in turn_data.get("steps", [])
                ],
            )
            for turn_data in data.get("turns", [])
        ]
        retrieval_contexts = [
            RetrievalContext(
                doc_id=item["doc_id"],
                text=item["text"],
                score=item["score"],
                collection=item["collection"],
                query=item["query"],
                turn_index=item["turn_index"],
                timestamp=datetime.fromisoformat(item["timestamp"]),
            )
            for item in data.get("retrieval_contexts", [])
        ]
        return cls(turns=turns, retrieval_contexts=retrieval_contexts)

    def save_history_to_json(self, path: str = "./conversations/history.json") -> None:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_history_from_json(self, path: str = "./conversations/history.json") -> list[Turn]:
        file = Path(path)
        if not file.exists():
            self.turns = []
            self.retrieval_contexts = []
            return self.turns

        loaded_context = self.from_dict(json.loads(file.read_text(encoding="utf-8")))
        self.turns = loaded_context.turns
        self.retrieval_contexts = loaded_context.retrieval_contexts
        return self.turns
