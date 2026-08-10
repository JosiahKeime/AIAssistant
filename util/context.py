from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path
from typing import Any

from dataclasses import dataclass, field

import config
from util.embedder import Embedder
from .logging_utils import _serialize_messages, logger

TERMINAL_STOP_REASONS = {"end_turn", "max_tokens", "stop_sequence", "refusal"}

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
            b.text for b in last.assistant_content
            if getattr(b, "type", None) == "text"
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

    def save_history_to_json(self, path: str = "./converstations/history.json"):
        path_obj = Path(path) + time.now().strftime("_%Y%m%d_%H%M%S.json")
        with open(path_obj, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_history_from_json(self, path: str = "./converstations/history.json") -> list:
        file = Path(path)
        if not file.exists():
            return []
        with open(file, "r") as f:
            loaded_history = json.load(f)
            self.history = list(loaded_history)
            return self.history
