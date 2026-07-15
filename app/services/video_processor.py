import json
import uuid
from pathlib import Path

from app.config import OUTPUT_DIR, UPLOAD_DIR
from app.models.schemas import (
    GenerateQuestionsResponse,
    TranscriptSegment,
)
from app.pipeline.audio_extractor import extract_audio
from app.pipeline.question_generator import generate_questions
from app.pipeline.transcriber import (
    filter_segments_by_watch_time,
    segments_to_text,
    transcribe_audio,
)


def _meta_path(video_id: str) -> Path:
    """
    Construct the path to the metadata file for a given video.

    Args:
        video_id (str):
            Unique identifier of the uploaded video.

    Returns:
        Path:
            Path to the `meta.json` file containing video metadata.
    """
    return OUTPUT_DIR / video_id / "meta.json"


def _transcript_path(video_id: str) -> Path:
    """
    Construct the path to the transcript file for a given video.

    Args:
        video_id (str):
            Unique identifier of the uploaded video.

    Returns:
        Path:
            Path to the `transcript.json` file containing transcript
            segments with timestamps.
    """
    return OUTPUT_DIR / video_id / "transcript.json"


def save_upload(video_bytes: bytes, filename: str) -> str:
    """
    Save an uploaded video file to disk and create its metadata.

    A unique video identifier is generated for each upload. The video
    file is stored under the uploads directory, and metadata describing
    the file is written to `meta.json`.

    Args:
        video_bytes (bytes):
            Raw bytes of the uploaded video.

        filename (str):
            Original name of the uploaded file.

    Returns:
        str:
            Generated video identifier.

    Example:
        >>> video_id = save_upload(data, "lecture.mp4")
        >>> print(video_id)
        'a1b2c3d4'
    """
    video_id = str(uuid.uuid4())[:8]

    # Create upload directory.
    video_dir = UPLOAD_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    # Save the uploaded file.
    video_path = video_dir / filename
    video_path.write_bytes(video_bytes)

    # Store metadata for future processing.
    meta = {
        "video_id": video_id,
        "filename": filename,
        "path": str(video_path),
    }

    out_dir = OUTPUT_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    _meta_path(video_id).write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    return video_id


def process_video(video_id: str) -> list[TranscriptSegment]:
    """
    Execute the complete video processing pipeline.

    Pipeline Steps:
        1. Load video metadata.
        2. Extract audio from the video.
        3. Transcribe audio using Faster-Whisper.
        4. Save transcript segments to disk.

    Args:
        video_id (str):
            Unique identifier of the uploaded video.

    Returns:
        list[TranscriptSegment]:
            Timestamped transcript segments extracted from the video.

    Raises:
        FileNotFoundError:
            If the metadata file or source video cannot be found.

        RuntimeError:
            If audio extraction or transcription fails.
    """
    # Load metadata.
    meta = json.loads(
        _meta_path(video_id).read_text(encoding="utf-8")
    )

    video_path = Path(meta["path"])
    out_dir = OUTPUT_DIR / video_id

    # Extract audio from video.
    audio_path = out_dir / "audio.wav"
    extract_audio(video_path, audio_path)

    # Generate transcript.
    segments = transcribe_audio(audio_path)

    # Persist transcript for reuse.
    _transcript_path(video_id).write_text(
        json.dumps(
            [segment.model_dump() for segment in segments],
            indent=2,
        ),
        encoding="utf-8",
    )

    return segments


def load_transcript(video_id: str) -> list[TranscriptSegment]:
    """
    Load a previously generated transcript from disk.

    Args:
        video_id (str):
            Unique identifier of the video.

    Returns:
        list[TranscriptSegment]:
            List of transcript segments. Returns an empty list if
            no transcript file exists.
    """
    path = _transcript_path(video_id)

    if not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))

    return [
        TranscriptSegment(**item)
        for item in raw
    ]


def generate_for_watch_time(
    video_id: str,
    watch_time_seconds: float,
    num_questions: int = 5,
) -> GenerateQuestionsResponse:
    """
    Generate questions based on the portion of a video watched by a user.

    If a transcript does not already exist for the specified video,
    the video processing pipeline is executed automatically.

    Workflow:
        1. Load or generate transcript.
        2. Filter transcript segments up to the user's watch time.
        3. Combine transcript text.
        4. Generate comprehension questions.
        5. Return a structured API response.

    Args:
        video_id (str):
            Unique identifier of the video.

        watch_time_seconds (float):
            Number of seconds watched by the user.

        num_questions (int, optional):
            Maximum number of questions to generate. Defaults to 5.

    Returns:
        GenerateQuestionsResponse:
            Response object containing:
                - Video ID
                - Watch duration
                - Watched transcript text
                - Transcript segments used
                - Generated questions

    Example:
        >>> response = generate_for_watch_time(
        ...     video_id="a1b2c3d4",
        ...     watch_time_seconds=600,
        ...     num_questions=3
        ... )
        >>> len(response.questions)
        3
    """
    # Load existing transcript if available.
    segments = load_transcript(video_id)

    # Process the video if no transcript exists.
    if not segments:
        segments = process_video(video_id)

    # Filter transcript by watch duration.
    watched = filter_segments_by_watch_time(
        segments,
        watch_time_seconds,
    )

    # Convert transcript segments into plain text.
    transcript_text = segments_to_text(watched)

    # Generate questions from watched content.
    questions = generate_questions(
        watched,
        num_questions=num_questions,
    )

    return GenerateQuestionsResponse(
        video_id=video_id,
        watch_time_seconds=watch_time_seconds,
        watched_transcript=transcript_text,
        segments_used=watched,
        questions=questions,
    )