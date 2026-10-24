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
