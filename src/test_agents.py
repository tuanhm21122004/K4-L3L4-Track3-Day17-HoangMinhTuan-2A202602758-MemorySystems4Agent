from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    extract_profile_updates,
)
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated LabConfig for testing."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    data_dir = Path(__file__).resolve().parent.parent / "data"

    dummy_model = ProviderConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
        api_key=None,
    )

    return LabConfig(
        base_dir=tmp_path,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=150,  # low threshold so compaction triggers easily
        compact_keep_messages=2,
        model=dummy_model,
        judge_model=dummy_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, read, edited, and monitored for file size."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "dungct_test"

    # 1. Non-existent file reads as empty with 0 bytes
    assert store.read_text(user_id) == ""
    assert store.file_size(user_id) == 0

    # 2. Write markdown profile
    initial_content = "# User Profile: dungct_test\n\n- **Name**: DungCT\n- **Location**: Danang\n"
    path = store.write_text(user_id, initial_content)
    assert path.is_file()
    assert store.read_text(user_id) == initial_content
    assert store.file_size(user_id) > 0

    # 3. Edit text inside User.md
    edit_success = store.edit_text(user_id, "Danang", "Hue")
    assert edit_success is True
    updated_content = store.read_text(user_id)
    assert "Hue" in updated_content
    assert "Danang" not in updated_content

    # 4. Editing non-matching text returns False
    failed_edit = store.edit_text(user_id, "Non-existent", "Something")
    assert failed_edit is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction when threshold is reached."""
    # Low threshold: 80 tokens, keep last 2 messages
    manager = CompactMemoryManager(threshold_tokens=80, keep_messages=2)
    thread_id = "thread_test"

    # Append short turns
    manager.append(thread_id, "user", "Chào bạn, đây là tin ngắn 1.")
    manager.append(thread_id, "assistant", "Chào bạn, tôi đã ghi nhận.")
    assert manager.compaction_count(thread_id) == 0

    # Append a long turn pushing token count well beyond threshold
    long_msg = "Đây là một đoạn văn bản rất dài về chương trình Artemis III của NASA " * 10
    manager.append(thread_id, "user", long_msg)

    # Compaction must have been triggered
    assert manager.compaction_count(thread_id) >= 1

    ctx = manager.context(thread_id)
    assert ctx["summary"] != ""
    assert len(ctx["messages"]) <= 2


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify Advanced Agent remembers facts across fresh sessions while Baseline does not."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "dungct_recall_test"
    thread_session_1 = "session-1"
    thread_session_2 = "session-2-fresh"

    # Session 1: User introduces facts
    intro_message = "Chào bạn, mình tên là DũngCT. Đồ uống yêu thích là cà phê sữa đá."
    baseline.reply(user_id, thread_session_1, intro_message)
    advanced.reply(user_id, thread_session_1, intro_message)

    # Session 2 (fresh thread): Query facts
    query = "Mình tên gì và đồ uống yêu thích là gì?"
    baseline_answer = baseline.reply(user_id, thread_session_2, query)["answer"]
    advanced_answer = advanced.reply(user_id, thread_session_2, query)["answer"]

    # Baseline MUST NOT recall across threads (score 0)
    assert "DũngCT" not in baseline_answer
    assert "cà phê sữa đá" not in baseline_answer

    # Advanced MUST recall across threads using User.md
    assert "DũngCT" in advanced_answer
    assert "cà phê sữa đá" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Verify compact memory significantly reduces prompt token load on long threads."""
    config = make_config(tmp_path)
    config.compact_threshold_tokens = 250
    config.compact_keep_messages = 2

    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "dungct_stress_test"
    thread_id = "long_thread_benchmark"

    # Send 14 long turns
    long_news = (
        "Chiếc máy bay X-59 lần đầu bay siêu thanh đạt tốc độ Mach 1.1 "
        "nhằm giảm thiểu tiếng nổ sonic boom để bảo vệ khu dân cư. "
        "Kế hoạch năng lượng sạch British Columbia dự kiến nhu cầu tăng 20% vào năm 2030. "
    )

    for i in range(14):
        msg = f"Lượt {i}: {long_news}"
        baseline.reply(user_id, thread_id, msg)
        advanced.reply(user_id, thread_id, msg)

    # Advanced must trigger compaction
    assert advanced.compaction_count(thread_id) > 0

    baseline_prompt = baseline.prompt_token_usage(thread_id)
    advanced_prompt = advanced.prompt_token_usage(thread_id)

    # Advanced prompt load must be lower than baseline's uncompressed context accumulation
    assert advanced_prompt < baseline_prompt, (
        f"Advanced prompt ({advanced_prompt}) should be less than Baseline ({baseline_prompt})"
    )


# ==============================================================================
# BONUS TESTS: Full points for Bonus Criteria (Rubric 90-100)
# ==============================================================================

def test_confidence_threshold_and_noise_filtering(tmp_path: Path) -> None:
    """Bonus Test 1: Verify Confidence Threshold filters out jokes and noise."""
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    user_id = "bonus_user"
    thread_id = "t1"

    # Initial facts
    advanced.reply(user_id, thread_id, "Mình tên là DũngCT, đang làm MLOps engineer ở Huế.")

    # Joke message: "chuyển sang product manager ... chỉ là câu đùa"
    joke_msg = "Có lúc mình đùa rằng chuyển sang product manager, nhưng đó chỉ là câu đùa. Nghề hiện tại vẫn là MLOps engineer."
    advanced.reply(user_id, thread_id, joke_msg)

    # Noise message: "Hà Nội chỉ là nơi mình vừa bay ra họp"
    noise_msg = "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày chứ không phải nơi ở hiện tại."
    advanced.reply(user_id, thread_id, noise_msg)

    facts = advanced.profile_store.load_facts(user_id)
    # Must preserve real profession and location, ignoring noise and jokes
    assert facts.get("current_profession") == "MLOps engineer"
    assert facts.get("current_profession") != "product manager"
    assert facts.get("current_location") == "Huế"
    assert facts.get("current_location") != "Hà Nội"


def test_conflict_handling_and_correction(tmp_path: Path) -> None:
    """Bonus Test 2: Verify Conflict Handling overrides obsolete facts cleanly."""
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    user_id = "conflict_user"
    thread_id = "t1"

    # Step 1: Initial location Đà Nẵng, backend engineer
    advanced.reply(user_id, thread_id, "Mình tên là DũngCT, ở Đà Nẵng và làm backend engineer.")
    facts1 = advanced.profile_store.load_facts(user_id)
    assert facts1.get("current_location") == "Đà Nẵng"
    assert facts1.get("current_profession") == "backend engineer"

    # Step 2: Correction to Huế and MLOps engineer
    correction_msg = (
        "À mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa. "
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer."
    )
    advanced.reply(user_id, thread_id, correction_msg)

    facts2 = advanced.profile_store.load_facts(user_id)
    assert facts2.get("current_location") == "Huế"
    assert facts2.get("current_profession") == "MLOps engineer"


def test_entity_extraction_and_decay_tracking(tmp_path: Path) -> None:
    """Bonus Test 3: Verify structured entity extraction and memory provenance log."""
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    user_id = "entity_user"
    thread_id = "t1"

    # Comprehensive multi-entity turn
    msg = (
        "Mình tên là DũngCT. Món ăn yêu thích là mì Quảng. "
        "Mình nuôi một bé corgi tên Bơ. Đồ uống yêu thích là cà phê sữa đá. "
        "Mình thích Python, AI ứng dụng."
    )
    advanced.reply(user_id, thread_id, msg)

    facts = advanced.profile_store.load_facts(user_id)
    assert facts.get("name") == "DũngCT"
    assert facts.get("favorite_food") == "mì Quảng"
    assert facts.get("pet") == "corgi (tên Bơ)"
    assert facts.get("favorite_drink") == "cà phê sữa đá"
    assert "Python" in facts.get("technical_interests", "")
    assert "AI" in facts.get("technical_interests", "")

    # Profile file must be formatted and written to disk
    profile_md = advanced.profile_store.read_text(user_id)
    assert "## Core Identity" in profile_md
    assert "## Preferences & Habits" in profile_md


def test_baseline_never_remembers_across_threads(tmp_path: Path) -> None:
    """Bonus Test 4: Verify Baseline Agent strictly adheres to in-thread amnesia."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    user_id = "strict_baseline_user"

    # Feed facts across 3 different sessions
    baseline.reply(user_id, "session-A", "Mình tên là DũngCT, nghề MLOps engineer.")
    baseline.reply(user_id, "session-B", "Đồ uống yêu thích của mình là cà phê sữa đá.")
    baseline.reply(user_id, "session-C", "Mình ở Đà Nẵng và nuôi corgi.")

    # In a completely new session-D, baseline must have zero knowledge of any past sessions
    res_d = baseline.reply(user_id, "session-D", "Nhắc lại tên, nghề, đồ uống, nơi ở và thú cưng của mình?")["answer"]
    assert "DũngCT" not in res_d
    assert "MLOps engineer" not in res_d
    assert "cà phê sữa đá" not in res_d
    assert "Đà Nẵng" not in res_d
    assert "corgi" not in res_d
    assert baseline.memory_file_size(user_id) == 0


def test_hierarchical_summary_compression(tmp_path: Path) -> None:
    """Bonus Test 5: Verify hierarchical summary merging bounds summary size across repeated compactions."""
    manager = CompactMemoryManager(threshold_tokens=50, keep_messages=2)
    thread_id = "heavy_compaction_thread"

    # Force 10 compactions with repeated distinct topics
    topics = [
        "NASA Artemis III roadmap milestone",
        "X-59 supersonic aircraft sonic boom test",
        "WMO El Nino climate probability",
        "BC energy clean plan",
        "Async Python review",
        "RAG and agent evaluation discussion",
        "Memory compaction stress testing",
    ]

    for i in range(15):
        topic = topics[i % len(topics)]
        manager.append(thread_id, "user", f"Thảo luận về {topic} với nội dung rất dài " * 5)
        manager.append(thread_id, "assistant", "Ghi nhận nội dung kỹ thuật.")

    assert manager.compaction_count(thread_id) >= 5
    ctx = manager.context(thread_id)
    summary = ctx.get("summary", "")

    # Summary must exist and be deduplicated / bounded (not 10 repeated headers)
    assert summary.count("Summary of previous turns:") <= 1
    # Max bullet count should be respected (<= 8 bullets)
    bullet_lines = [l for l in summary.splitlines() if l.strip().startswith("- ")]
    assert len(bullet_lines) <= 8

