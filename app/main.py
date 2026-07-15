from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.models.schemas import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
)
from app.services.video_processor import (
    generate_for_watch_time,
    save_upload,
)
from app.services.watch_tracker import WatchTracker

# Initialize the FastAPI application.
app = FastAPI(
    title="Video Question Generator POC",
    description=(
        "Generate comprehension questions based on how much "
        "of a video a user has watched."
    ),
    version="0.1.0",
)

# Configure static file serving.
STATIC_DIR = Path(__file__).parent / "static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

# Initialize watch time tracker.
tracker = WatchTracker()


@app.get("/", response_class=HTMLResponse)
async def index():
    """
    Serve the application's home page.

    Returns:
        HTMLResponse:
            Contents of the `index.html` file located in the static
            directory.

    Endpoint:
        GET /
    """
    return (STATIC_DIR / "index.html").read_text(
        encoding="utf-8"
    )


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for processing.

    Supported formats:
        - MP4
        - WEBM
        - MKV
        - AVI
        - MOV

    Args:
        file (UploadFile):
            Video file uploaded by the client.

    Returns:
        dict:
            JSON response containing:
                - video_id
                - filename

    Raises:
        HTTPException:
            - 400 if no file is provided.
            - 400 if the file format is unsupported.

    Endpoint:
        POST /api/upload
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided",
        )

    allowed = {
        ".mp4",
        ".webm",
        ".mkv",
        ".avi",
        ".mov",
    }

    ext = Path(file.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported format. "
                f"Use: {', '.join(sorted(allowed))}"
            ),
        )

    content = await file.read()

    video_id = save_upload(
        content,
        file.filename,
    )

    return {
        "video_id": video_id,
        "filename": file.filename,
    }


@app.get("/api/video/{video_id}")
async def serve_video(video_id: str):
    """
    Stream a previously uploaded video.

    Args:
        video_id (str):
            Unique identifier of the uploaded video.

    Returns:
        FileResponse:
            Video file streamed back to the client.

    Raises:
        HTTPException:
            - 404 if the metadata file does not exist.
            - 404 if the underlying video file is missing.

    Endpoint:
        GET /api/video/{video_id}
    """
    import json

    from app.config import OUTPUT_DIR

    meta_path = OUTPUT_DIR / video_id / "meta.json"

    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video not found",
        )

    meta = json.loads(
        meta_path.read_text(encoding="utf-8")
    )

    video_path = Path(meta["path"])

    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video file missing",
        )

    return FileResponse(
        video_path,
        media_type="video/mp4",
    )


@app.post("/api/watch/{video_id}")
async def update_watch_time(
    video_id: str,
    seconds: float = Form(...),
):
    """
    Update today's watch time for a specific video.

    This endpoint replaces the existing watch time with the supplied
    value rather than incrementing it.

    Args:
        video_id (str):
            Unique identifier of the video.

        seconds (float):
            Number of seconds watched by the user.

    Returns:
        dict:
            JSON response containing:
                - video_id
                - watch_time_seconds
                - date

    Endpoint:
        POST /api/watch/{video_id}
    """
    total = tracker.set_watch_time(
        video_id,
        seconds,
    )

    return {
        "video_id": video_id,
        "watch_time_seconds": total,
        "date": "today",
    }


@app.get("/api/watch/{video_id}")
async def get_watch_time(video_id: str):
    """
    Retrieve today's watch time for a video.

    Args:
        video_id (str):
            Unique identifier of the video.

    Returns:
        dict:
            JSON response containing:
                - video_id
                - watch_time_seconds

    Endpoint:
        GET /api/watch/{video_id}
    """
    return {
        "video_id": video_id,
        "watch_time_seconds": tracker.get_watch_time(
            video_id
        ),
    }


@app.post(
    "/api/generate/{video_id}",
    response_model=GenerateQuestionsResponse,
)
async def generate_questions_endpoint(
    video_id: str,
    body: GenerateQuestionsRequest,
):
    """
    Generate comprehension questions for the watched portion of a video.

    Workflow:
        1. Update today's watch time.
        2. Load or generate the transcript.
        3. Filter transcript segments based on watch duration.
        4. Generate questions from the watched content.
        5. Return a structured response.

    Args:
        video_id (str):
            Unique identifier of the uploaded video.

        body (GenerateQuestionsRequest):
            Request payload containing:
                - watch_time_seconds
                - num_questions

    Returns:
        GenerateQuestionsResponse:
            Contains:
                - Video ID
                - Watch duration
                - Watched transcript
                - Transcript segments used
                - Generated questions

    Raises:
        HTTPException:
            - 404 if the video does not exist.
            - 500 for unexpected processing errors.

    Endpoint:
        POST /api/generate/{video_id}
    """
    try:
        # Persist watch time.
        tracker.set_watch_time(
            video_id,
            body.watch_time_seconds,
        )

        # Generate questions.
        return generate_for_watch_time(
            video_id,
            body.watch_time_seconds,
            num_questions=body.num_questions,
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Video not found",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )