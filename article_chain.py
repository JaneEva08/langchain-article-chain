"""Цепочка запросов (LangChain) для генерации статьи из 4 звеньев.

    тема -> analysis_chain -> outline_chain -> draft_chain -> review_chain -> article.md

Каждое звено получает только те данные, которые нужны именно на его шаге.

Запуск:
    python article_chain.py "Как начать работать с Docker"
"""
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)


def setup_logger() -> None:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def get_llm() -> RunnableLambda:
    """Runnable, который вызывает OpenAI Chat Completions напрямую.

    Переменные окружения: OPENAI_API_KEY (обязательно), OPENAI_BASE_URL,
    OPENAI_MODEL или AI_MODEL (по умолчанию gpt-4o-mini), TEMPERATURE (по умолчанию 0.3).
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    model = os.getenv("OPENAI_MODEL") or os.getenv("AI_MODEL") or "gpt-4o-mini"
    try:
        temperature = float(os.getenv("TEMPERATURE", "0.3"))
    except ValueError:
        temperature = 0.3
    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)

    roles = {"system": "system", "human": "user", "ai": "assistant"}

    def _invoke(prompt_value: Any) -> str:
        messages = [{"role": roles.get(m.type, "user"), "content": str(m.content)}
                    for m in prompt_value.to_messages()]
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        return resp.choices[0].message.content or ""

    logger.debug("OpenAI client model=%s temperature=%.1f", model, temperature)
    return RunnableLambda(_invoke)


LANG_RULE = "Write all text values in the same language as the article topic. "


def _chain(llm, system: str, human: str):
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    return prompt | llm | StrOutputParser()


# --- Звено 1: анализ задания ------------------------------------------------
def build_analysis_chain(llm):
    system = (
        f"You are a senior editor. {LANG_RULE}Analyze the article topic and output JSON with keys:\n"
        "- goal: what the reader should get from the article\n"
        "- audience: who the reader is and their level\n"
        "- key_questions: list of 3-6 questions the article must answer\n"
        "- tone: style of writing\n"
        "Return ONLY valid minified JSON. No code fences."
    )
    return _chain(llm, system, "Topic: {topic}")


# --- Звено 2: структура статьи ----------------------------------------------
def build_outline_chain(llm):
    system = (
        f"You are a content strategist. {LANG_RULE}Based on the analysis, design the article outline.\n"
        "Output JSON with keys:\n"
        "- title: article title\n"
        "- sections: list of {{heading, key_points}} where key_points is a list of strings\n"
        "- conclusion: one sentence with the main takeaway\n"
        "Return ONLY valid minified JSON. No code fences."
    )
    return _chain(llm, system, "Analysis: {analysis_json}")


# --- Звено 3: написание черновика -------------------------------------------
def build_draft_chain(llm):
    system = (
        "You are a professional writer. Write the full article in Markdown in the same language as "
        "the topic, strictly following "
        "the outline: one H2 per section, cover every key point, finish with the conclusion. "
        "Match the tone and audience from the analysis. "
        "Return ONLY the Markdown article. No code fences around it."
    )
    human = "Analysis: {analysis_json}\nOutline: {outline_json}\nWrite the article now."
    return _chain(llm, system, human)


# --- Звено 4: ревью ---------------------------------------------------------
def build_review_chain(llm):
    system = (
        "You are a strict copy editor. Keep the article language. Fix the listed issues and improve clarity without "
        "changing the structure or facts. Return ONLY the corrected Markdown article."
    )
    human = "Issues: {issues}\n\nArticle:\n{article}"
    return _chain(llm, system, human)


# --- Вспомогательные функции ------------------------------------------------
def strip_code_fences(text: str) -> str:
    """Снять ОДНУ внешнюю обёртку ```...``` вокруг всего ответа.

    Блоки кода внутри статьи не трогаем: они нужны читателю.
    """
    m = re.match(r"^\s*```[\w+-]*[ \t]*\n(.*)\n\s*```\s*$", text, flags=re.DOTALL)
    return m.group(1) if m else text


def parse_json(text: str) -> Dict[str, Any]:
    text = strip_code_fences(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def minimal_review(article: str, outline: Dict[str, Any]) -> List[str]:
    """Локальная проверка: есть ли заголовок, все разделы и достаточный объём."""
    issues: List[str] = []
    if not re.search(r"^# .+", article, flags=re.MULTILINE):
        issues.append("Missing H1 title")
    for section in outline.get("sections", []):
        heading = section.get("heading", "") if isinstance(section, dict) else str(section)
        if heading and heading.lower() not in article.lower():
            issues.append(f"Missing section: {heading}")
    try:
        min_words = int(os.getenv("MIN_WORDS", "150"))
    except ValueError:
        min_words = 150
    if len(article.split()) < min_words:
        issues.append(f"Article is too short (< {min_words} words)")
    return issues


_TRANSLIT = dict(zip(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r",
     "s", "t", "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"],
))


def slugify(text: str) -> str:
    """Имя файла из темы: кириллица транслитерируется, остальное -> '_'."""
    text = "".join(_TRANSLIT.get(ch, ch) for ch in text.lower())
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "article"


def unique_path(path: Path) -> Path:
    """Не затирать прошлый результат: article.md, article_2.md, ..."""
    if not path.exists():
        return path
    n = 2
    while path.with_name(f"{path.stem}_{n}{path.suffix}").exists():
        n += 1
    return path.with_name(f"{path.stem}_{n}{path.suffix}")


def run_chain(topic: str, out_dir: Path, llm=None) -> Path:
    llm = llm or get_llm()

    logger.info("Running analysis chain")
    analysis = parse_json(build_analysis_chain(llm).invoke({"topic": topic}))
    analysis_json = json.dumps(analysis, ensure_ascii=False)

    logger.info("Running outline chain")
    outline = parse_json(build_outline_chain(llm).invoke({"analysis_json": analysis_json}))
    outline_json = json.dumps(outline, ensure_ascii=False)

    logger.info("Running draft chain")
    article = strip_code_fences(build_draft_chain(llm).invoke(
        {"analysis_json": analysis_json, "outline_json": outline_json})).strip()

    logger.info("Running review chain")
    issues = minimal_review(article, outline)
    review_used = False
    if issues:
        logger.warning("Review found issues: %s", "; ".join(issues))
        fixed = strip_code_fences(build_review_chain(llm).invoke(
            {"issues": "\n".join(issues), "article": article})).strip()
        if fixed and len(minimal_review(fixed, outline)) < len(issues):
            article = fixed
            review_used = True
    else:
        logger.info("Review passed with no issues")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = unique_path(out_dir / f"{slugify(topic)[:40]}.md")
    out_path.write_text(article, encoding="utf-8")
    # Сырые результаты звеньев: пригодятся для отчёта и отладки (в output/, он в .gitignore)
    steps = {"topic": topic, "analysis": analysis, "outline": outline,
             "review_issues": issues, "review_fix_applied": review_used}
    out_path.with_suffix(".steps.json").write_text(
        json.dumps(steps, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Written article: %s", out_path)
    return out_path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an article via a 4-step LangChain chain.")
    parser.add_argument("topic", help="Article topic")
    parser.add_argument("--out-dir", default="output", help="Directory for the result (default: output)")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    setup_logger()
    try:
        print(run_chain(args.topic, Path(args.out_dir)).resolve())
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
