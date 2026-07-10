from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.models.schemas import GenerateQuestionsRequest, GenerateQuestionsResponse
from app.services.video_processor import generate_for_watch_time, save_upload
from app.services.watch_tracker import WatchTracker

app = FastAPI(
    title="Video Question Generator POC",
    description="Generate questions based on how much video was watched today",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

tracker = WatchTracker()


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file provided")

    allowed = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format. Use: {', '.join(allowed)}")

    content = await file.read()
    video_id = save_upload(content, file.filename)
    return {"video_id": video_id, "filename": file.filename}


@app.get("/api/video/{video_id}")
async def serve_video(video_id: str):
    import json

    from app.config import OUTPUT_DIR

    meta_path = OUTPUT_DIR / video_id / "meta.json"
    if not meta_path.exists():
        raise HTTPException(404, "Video not found")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    video_path = Path(meta["path"])
    if not video_path.exists():
        raise HTTPException(404, "Video file missing")

    return FileResponse(video_path, media_type="video/mp4")


@app.post("/api/watch/{video_id}")
async def update_watch_time(video_id: str, seconds: float = Form(...)):
    total = tracker.set_watch_time(video_id, seconds)
    return {"video_id": video_id, "watch_time_seconds": total, "date": "today"}


@app.get("/api/watch/{video_id}")
async def get_watch_time(video_id: str):
    return {"video_id": video_id, "watch_time_seconds": tracker.get_watch_time(video_id)}


@app.post("/api/generate/{video_id}", response_model=GenerateQuestionsResponse)
async def generate_questions_endpoint(
    video_id: str,
    body: GenerateQuestionsRequest,
):
    try:
        tracker.set_watch_time(video_id, body.watch_time_seconds)
        return generate_for_watch_time(
            video_id,
            body.watch_time_seconds,
            num_questions=body.num_questions,
        )
    except FileNotFoundError:
        raise HTTPException(404, "Video not found")
    except Exception as exc:
        raise HTTPException(500, str(exc))
