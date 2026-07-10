import json
import uuid
from pathlib import Path

from app.config import OUTPUT_DIR, UPLOAD_DIR
from app.models.schemas import GenerateQuestionsResponse, TranscriptSegment
from app.pipeline.audio_extractor import extract_audio
from app.pipeline.question_generator import generate_questions
from app.pipeline.transcriber import (
    filter_segments_by_watch_time,
    segments_to_text,
    transcribe_audio,
)


def _meta_path(video_id: str) -> Path:
    return OUTPUT_DIR / video_id / "meta.json"


def _transcript_path(video_id: str) -> Path:
    return OUTPUT_DIR / video_id / "transcript.json"


def save_upload(video_bytes: bytes, filename: str) -> str:
    video_id = str(uuid.uuid4())[:8]
    video_dir = UPLOAD_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / filename
    video_path.write_bytes(video_bytes)

    meta = {"video_id": video_id, "filename": filename, "path": str(video_path)}
    out_dir = OUTPUT_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _meta_path(video_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return video_id


def process_video(video_id: str) -> list[TranscriptSegment]:
    """Full pipeline: video → audio → transcript with timestamps."""
    meta = json.loads(_meta_path(video_id).read_text(encoding="utf-8"))
    video_path = Path(meta["path"])
    out_dir = OUTPUT_DIR / video_id

    audio_path = out_dir / "audio.wav"
    extract_audio(video_path, audio_path)

    segments = transcribe_audio(audio_path)
    _transcript_path(video_id).write_text(
        json.dumps([s.model_dump() for s in segments], indent=2),
        encoding="utf-8",
    )
    return segments


def load_transcript(video_id: str) -> list[TranscriptSegment]:
    path = _transcript_path(video_id)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TranscriptSegment(**item) for item in raw]


def generate_for_watch_time(
    video_id: str,
    watch_time_seconds: float,
    num_questions: int = 5,
) -> GenerateQuestionsResponse:
    segments = load_transcript(video_id)
    if not segments:
        segments = process_video(video_id)

    watched = filter_segments_by_watch_time(segments, watch_time_seconds)
    transcript_text = segments_to_text(watched)
    questions = generate_questions(watched, num_questions=num_questions)

    return GenerateQuestionsResponse(
        video_id=video_id,
        watch_time_seconds=watch_time_seconds,
        watched_transcript=transcript_text,
        segments_used=watched,
        questions=questions,
    )
