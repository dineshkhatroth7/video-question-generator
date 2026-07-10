from pathlib import Path

from faster_whisper import WhisperModel

from app.config import WHISPER_MODEL
from app.models.schemas import TranscriptSegment

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def transcribe_audio(audio_path: Path) -> list[TranscriptSegment]:
    """Convert audio to text with timestamps using faster-whisper (free, local)."""
    model = _get_model()
    segments, _info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
    )

    return [
        TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip())
        for seg in segments
        if seg.text.strip()
    ]


def filter_segments_by_watch_time(
    segments: list[TranscriptSegment],
    watch_time_seconds: float,
) -> list[TranscriptSegment]:
    """Keep only transcript content up to today's watch time."""
    return [seg for seg in segments if seg.start < watch_time_seconds]


def segments_to_text(segments: list[TranscriptSegment]) -> str:
    return " ".join(seg.text for seg in segments).strip()
