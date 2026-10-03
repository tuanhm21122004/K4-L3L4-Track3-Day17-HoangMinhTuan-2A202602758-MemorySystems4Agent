from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Baseline Agent (Agent A).

    Characteristics:
    - Within-session / in-thread short-term memory only.
    - No persistent storage (no User.md).
    - Completely forgets facts across new sessions or new thread IDs.
    - No compact memory: context grows linearly with every turn in the thread.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline and self.config.model.api_key:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def _ensure_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Process incoming user message and return response with token metrics."""
        session = self._ensure_session(thread_id)

        # Route to offline deterministic path if forced or no live agent
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(thread_id, message)

        # Live LangChain / LangGraph path
        try:
            # Baseline uses live chat model without profile tools or persistent memory
            prompt_context = "\n".join(
                f"{m['role']}: {m['content']}" for m in session.messages
            ) + f"\nuser: {message}"
            prompt_tokens = estimate_tokens(prompt_context)
            session.prompt_tokens_processed += prompt_tokens

            session.messages.append({"role": "user", "content": message})
            response = self.langchain_agent.invoke(session.messages)
            answer_text = response.content if hasattr(response, "content") else str(response)

            agent_tokens = estimate_tokens(answer_text)
            session.token_usage += agent_tokens
            session.messages.append({"role": "assistant", "content": answer_text})

            return {
                "answer": answer_text,
                "content": answer_text,
                "tokens": agent_tokens,
                "prompt_tokens": session.prompt_tokens_processed,
            }
        except Exception:
            return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative agent-generated tokens for a thread or all threads."""
        if thread_id is not None:
            return self.sessions.get(thread_id, SessionState()).token_usage
        return sum(s.token_usage for s in self.sessions.values())

    def prompt_token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative prompt tokens processed for a thread or all threads."""
        if thread_id is not None:
            return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed
        return sum(s.prompt_tokens_processed for s in self.sessions.values())

    def compaction_count(self, thread_id: str | None = None) -> int:
        """Baseline agent has no compact memory; compactions are always 0."""
        return 0

    def memory_file_size(self, user_id: str) -> int:
        """Baseline agent does not write any persistent file; size is always 0."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline mode for baseline agent."""
        session = self._ensure_session(thread_id)

        # Add incoming user message to session
        session.messages.append({"role": "user", "content": message})

        # Context carried into this turn is the entire thread history so far (no compaction!)
        current_context = "\n".join(
            f"{m['role']}: {m['content']}" for m in session.messages
        )
        prompt_tokens = estimate_tokens(current_context)
        session.prompt_tokens_processed += prompt_tokens

        # Check if the query is asking about cross-session facts in a new thread
        is_recall_query = any(
            q in message.lower()
            for q in [
                "mình tên gì", "tên mình là gì", "đồ uống yêu thích", "món ăn yêu thích",
                "ở đâu", "làm nghề gì", "style trả lời", "con gì", "tóm tắt ngắn về mình",
                "đâu mới là nghề nghiệp", "sang thread mới rồi"
            ]
        )

        # Within-session memory check: did the user state this within THIS specific thread?
        user_turns_in_thread = [m["content"] for m in session.messages if m["role"] == "user"]
        has_in_thread_fact = len(user_turns_in_thread) > 1 and any(
            "mình tên là" in t.lower() or "đang làm" in t.lower() for t in user_turns_in_thread[:-1]
        )

        if is_recall_query and not has_in_thread_fact:
            # Baseline forgets across sessions! It has no User.md to recall facts from.
            reply_text = (
                "Xin chào! Vì đây là một phiên làm việc mới và tôi chỉ có bộ nhớ tạm thời trong từng thread "
                "(không có persistent profile User.md), tôi không thể nhớ tên, nghề nghiệp, nơi ở "
                "hay sở thích từ các phiên trước của bạn."
            )
        else:
            # Courteous in-thread acknowledgment
            reply_text = (
                f"Đã nhận thông tin trong phiên hiện tại ({thread_id}). "
                "Tôi đang duy trì ngữ cảnh trong thread này nhưng không lưu hồ sơ dài hạn."
            )

        reply_tokens = estimate_tokens(reply_text)
        session.token_usage += reply_tokens
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "answer": reply_text,
            "content": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": session.prompt_tokens_processed,
        }

    def _maybe_build_langchain_agent(self):
        """Construct a live ChatModel for baseline if provider configuration is valid."""
        return build_chat_model(self.config.model)
