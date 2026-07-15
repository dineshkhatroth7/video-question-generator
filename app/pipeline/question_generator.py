import re

from transformers import pipeline

from app.config import QUESTION_MODEL
from app.models.schemas import QuestionItem, TranscriptSegment

# Singleton instance of the Hugging Face question-generation pipeline.
_qg_pipeline = None


def _get_pipeline():
    """
    Lazily initialize and return the Hugging Face text generation pipeline.

    The pipeline is created only once and reused across subsequent calls
    to avoid repeatedly loading the model into memory, which can be an
    expensive operation.

    Returns:
        transformers.pipelines.Pipeline:
            Configured text-to-text generation pipeline for question
            generation using the model specified in `QUESTION_MODEL`.
    """
    global _qg_pipeline

    if _qg_pipeline is None:
        _qg_pipeline = pipeline(
            "text2text-generation",
            model=QUESTION_MODEL,
            max_length=64,
        )

    return _qg_pipeline


def _split_into_chunks(text: str, max_words: int = 80) -> list[str]:
    """
    Split a transcript into smaller chunks suitable for question generation.

    The transcript is first divided into sentences and then grouped into
    chunks containing approximately `max_words` words. This helps maintain
    context while ensuring the generated prompts remain within model limits.

    Args:
        text (str):
            Complete transcript text.

        max_words (int, optional):
            Maximum number of words per chunk. Defaults to 80.

    Returns:
        list[str]:
            List of transcript chunks. If no chunks are produced, the
            original text is returned as a single-element list.
    """
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
    """
    Generate a single question from a given text context.

    This function sends the provided context to the Hugging Face
    text-to-text generation model using the prompt format expected
    by T5-style question generation models.

    Args:
        context (str):
            Transcript text used as input for question generation.

    Returns:
        str:
            Generated question ending with a question mark.

    Example:
        >>> _generate_question(
        ...     "Python is a programming language used for web development."
        ... )
        'What is Python used for?'
    """
    pipe = _get_pipeline()

    # Limit prompt size to avoid exceeding model context limits.
    prompt = f"generate question: {context[:512]}"

    result = pipe(
        prompt,
        max_new_tokens=48,
        num_return_sequences=1,
    )

    question = result[0]["generated_text"].strip()

    # Ensure the output is formatted as a question.
    if not question.endswith("?"):
        question = question.rstrip(".") + "?"

    return question


def generate_questions(
    segments: list[TranscriptSegment],
    num_questions: int = 5,
) -> list[QuestionItem]:
    """
    Generate comprehension questions from transcript segments watched by a user.

    The function combines transcript segments into text chunks, generates
    questions for each chunk using a transformer model, and returns the
    resulting questions along with metadata such as timestamps and source
    transcript segments.

    If the model fails to generate a question, a fallback question is
    generated based on the segment's timestamp.

    Args:
        segments (list[TranscriptSegment]):
            List of transcript segments corresponding to the watched portion
            of the video.

        num_questions (int, optional):
            Maximum number of questions to generate. Defaults to 5.

    Returns:
        list[QuestionItem]:
            A list of generated question objects containing:
                - question
                - answer hint
                - source transcript segment
                - segment timestamp

    Example:
        >>> questions = generate_questions(segments, num_questions=3)
        >>> len(questions)
        3
    """
    if not segments:
        return []

    # Combine all transcript text.
    full_text = " ".join(seg.text for seg in segments)

    # Split transcript into manageable chunks.
    chunks = _split_into_chunks(full_text)

    questions: list[QuestionItem] = []

    for i, chunk in enumerate(chunks):
        if len(questions) >= num_questions:
            break

        source_seg = segments[min(i, len(segments) - 1)]

        try:
            question = _generate_question(chunk)
        except Exception:
            # Fallback question if model inference fails.
            question = (
                f"What is the main idea discussed around "
                f"{source_seg.start:.0f}s?"
            )

        questions.append(
            QuestionItem(
                question=question,
                answer_hint=(
                    chunk[:200] + ("..." if len(chunk) > 200 else "")
                ),
                source_segment=source_seg.text,
                timestamp_start=source_seg.start,
            )
        )

    return questions