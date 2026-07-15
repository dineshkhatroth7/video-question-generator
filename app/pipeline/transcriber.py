from pathlib import Path

from faster_whisper import WhisperModel

from app.config import WHISPER_MODEL
from app.models.schemas import TranscriptSegment

# Singleton instance of the Faster-Whisper model.
_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """
    Lazily initialize and return the Faster-Whisper model.

    The model is loaded only once and reused throughout the application's
    lifecycle to minimize startup overhead and reduce memory consumption.

    The model configuration is obtained from `WHISPER_MODEL` and is loaded
    using CPU execution with INT8 quantization for faster inference and
    lower resource utilization.

    Returns:
        WhisperModel:
            Initialized Faster-Whisper model instance.
    """
    global _model

    if _model is None:
        _model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
        )

    return _model


def transcribe_audio(audio_path: Path) -> list[TranscriptSegment]:
    """
    Transcribe an audio file into timestamped transcript segments.

    This function uses Faster-Whisper to perform automatic speech
    recognition (ASR) on the provided audio file. Voice Activity
    Detection (VAD) is enabled to filter out silent portions of
    the recording.

    Args:
        audio_path (Path):
            Path to the input audio file (typically a WAV file).

    Returns:
        list[TranscriptSegment]:
            A list of transcript segments containing:
                - Start timestamp (seconds)
                - End timestamp (seconds)
                - Transcribed text

    Raises:
        Exception:
            Propagates any exceptions raised during model loading or
            transcription.

    Example:
        >>> segments = transcribe_audio(Path("lecture.wav"))
        >>> segments[0].text
        'Welcome to today's lecture on machine learning.'
    """
    model = _get_model()

    segments, _info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
    )

    return [
        TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
        )
        for seg in segments
        if seg.text.strip()
    ]


def filter_segments_by_watch_time(
    segments: list[TranscriptSegment],
    watch_time_seconds: float,
) -> list[TranscriptSegment]:
    """
    Filter transcript segments based on the user's watch time.

    Only segments whose start time is less than the specified watch
    duration are included. This allows the application to generate
    questions only for content the user has actually watched.

    Args:
        segments (list[TranscriptSegment]):
            Complete list of transcript segments.

        watch_time_seconds (float):
            Number of seconds watched by the user.

    Returns:
        list[TranscriptSegment]:
            Filtered transcript segments corresponding to the watched
            portion of the video.

    Example:
        >>> watched_segments = filter_segments_by_watch_time(
        ...     segments,
        ...     watch_time_seconds=300
        ... )
    """
    return [
        seg
        for seg in segments
        if seg.start < watch_time_seconds
    ]


def segments_to_text(segments: list[TranscriptSegment]) -> str:
    """
    Convert transcript segments into a single text string.

    This utility function concatenates the text from all transcript
    segments into a single whitespace-separated string, making it
    suitable for downstream processing such as summarization or
    question generation.

    Args:
        segments (list[TranscriptSegment]):
            List of transcript segments.

    Returns:
        str:
            Combined transcript text with leading and trailing
            whitespace removed.

    Example:
        >>> text = segments_to_text(segments)
        >>> print(text)
        'This is the first sentence. This is the second sentence.'
    """
    return " ".join(seg.text for seg in segments).strip()