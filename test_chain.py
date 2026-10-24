"""Проверка цепочки без обращения к OpenAI: LLM заменён заглушкой."""
import json
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from article_chain import run_chain

ANALYSIS = {"goal": "g", "audience": "a", "key_questions": ["q"], "tone": "t"}
OUTLINE = {"title": "T", "sections": [{"heading": "Intro", "key_points": ["x"]},
                                      {"heading": "Details", "key_points": ["y"]}],
           "conclusion": "c"}
ARTICLE = ("# T\n\n## Intro\n\n" + "word " * 100 + "\n\n## Details\n\n" + "word " * 100).strip()


def fake_llm_factory(calls):
    def _invoke(prompt_value):
        system = prompt_value.to_messages()[0].content
        if "senior editor" in system:
            calls.append("analysis")
            return json.dumps(ANALYSIS)
        if "content strategist" in system:
            calls.append("outline")
            return "```json\n" + json.dumps(OUTLINE) + "\n```"
        if "professional writer" in system:
            calls.append("draft")
            return ARTICLE
        calls.append("review")
        return ARTICLE
    return RunnableLambda(_invoke)


def test_chain_runs_four_steps_in_order(tmp_path: Path):
    calls = []
    out = run_chain("Test topic", tmp_path, llm=fake_llm_factory(calls))
    assert calls == ["analysis", "outline", "draft"]  # review-звено вызывается только при проблемах
    assert out.read_text(encoding="utf-8").startswith("# T")


def test_review_chain_fixes_broken_draft(tmp_path: Path, monkeypatch):
    calls = []
    llm = fake_llm_factory(calls)

    def broken(prompt_value):
        system = prompt_value.to_messages()[0].content
        if "professional writer" in system:
            calls.append("draft")
            return "short text"
        return llm.invoke(prompt_value)

    out = run_chain("Test topic", tmp_path, llm=RunnableLambda(broken))
    assert calls[-2:] == ["draft", "review"]
    assert out.read_text(encoding="utf-8") == ARTICLE


# --- регрессии по итогам ревью -------------------------------------------------

def test_strip_code_fences_keeps_inner_code_blocks():
    from article_chain import strip_code_fences
    article = "# Docker\n\n## Старт\n\n```bash\ndocker run hello-world\n```\n\nГотово."
    assert strip_code_fences(article) == article


def test_strip_code_fences_removes_outer_wrapper_only():
    from article_chain import strip_code_fences
    inner = "# T\n\n```bash\nls\n```\n\nКонец."
    assert strip_code_fences("```markdown\n" + inner + "\n```") == inner
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_slugify_transliterates_cyrillic():
    from article_chain import slugify
    assert slugify("Как начать работать с Docker") == "kak_nachat_rabotat_s_docker"
    assert slugify("???") == "article"


def test_second_run_does_not_overwrite_first(tmp_path: Path):
    first = run_chain("Тема статьи", tmp_path, llm=fake_llm_factory([]))
    second = run_chain("Тема статьи", tmp_path, llm=fake_llm_factory([]))
    assert first != second
    assert first.exists() and second.exists()
    assert first.name == "tema_stati.md" and second.name == "tema_stati_2.md"


def test_steps_file_is_saved(tmp_path: Path):
    out = run_chain("Test topic", tmp_path, llm=fake_llm_factory([]))
    steps = json.loads(out.with_suffix(".steps.json").read_text(encoding="utf-8"))
    assert steps["analysis"] == ANALYSIS and steps["outline"] == OUTLINE
    assert steps["review_issues"] == [] and steps["review_fix_applied"] is False


def test_prompts_ask_for_topic_language():
    from article_chain import build_analysis_chain, build_outline_chain
    seen = []
    def spy(prompt_value):
        seen.append(prompt_value.to_messages()[0].content)
        return "{}"
    build_analysis_chain(RunnableLambda(spy)).invoke({"topic": "x"})
    build_outline_chain(RunnableLambda(spy)).invoke({"analysis_json": "{}"})
    assert all("same language" in s for s in seen)


def test_min_words_env_triggers_review(tmp_path: Path, monkeypatch):
    """MIN_WORDS позволяет вызвать review-звено на нормальной статье (для демонстрации)."""
    monkeypatch.setenv("MIN_WORDS", "1000")
    calls = []
    out = run_chain("Test topic", tmp_path, llm=fake_llm_factory(calls))
    assert calls[-1] == "review"
    steps = json.loads(out.with_suffix(".steps.json").read_text(encoding="utf-8"))
    assert steps["review_issues"] == ["Article is too short (< 1000 words)"]
    assert steps["review_fix_applied"] is False  # правка не уменьшила число замечаний, не принята
