import re

from transformers import pipeline

from app.config import QUESTION_MODEL
from app.models.schemas import QuestionItem, TranscriptSegment

_qg_pipeline = None


def _get_pipeline():
    global _qg_pipeline
    if _qg_pipeline is None:
        _qg_pipeline = pipeline(
            "text2text-generation",
            model=QUESTION_MODEL,
            max_length=64,
        )
    return _qg_pipeline


def _split_into_chunks(text: str, max_words: int = 80) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        if not sentence.strip():
            continue
        current.append(sentence.strip())
        if len(" ".join(current).split()) >= max_words:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks or [text]


def _generate_question(context: str) -> str:
    pipe = _get_pipeline()
    prompt = f"generate question: {context[:512]}"
    result = pipe(prompt, max_new_tokens=48, num_return_sequences=1)
    question = result[0]["generated_text"].strip()
    if not question.endswith("?"):
        question = question.rstrip(".") + "?"
    return question


def generate_questions(
    segments: list[TranscriptSegment],
    num_questions: int = 5,
) -> list[QuestionItem]:
    """Generate comprehension questions from watched transcript segments."""
    if not segments:
        return []

    full_text = " ".join(seg.text for seg in segments)
    chunks = _split_into_chunks(full_text)
    questions: list[QuestionItem] = []

    for i, chunk in enumerate(chunks):
        if len(questions) >= num_questions:
            break

        source_seg = segments[min(i, len(segments) - 1)]
        try:
            question = _generate_question(chunk)
        except Exception:
            question = f"What is the main idea discussed around {source_seg.start:.0f}s?"

        questions.append(
            QuestionItem(
                question=question,
                answer_hint=chunk[:200] + ("..." if len(chunk) > 200 else ""),
                source_segment=source_seg.text,
                timestamp_start=source_seg.start,
            )
        )

    return questions
