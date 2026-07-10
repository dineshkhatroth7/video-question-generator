from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"

# Free local models (no API keys required)
WHISPER_MODEL = "base"  # tiny | base | small — base is a good POC balance
QUESTION_MODEL = "iarfmoose/t5-base-question-generator"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
