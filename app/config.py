from pathlib import Path

"""
Application configuration settings.

This module centralizes all application-level constants and paths used
throughout the Video Question Generator POC. It defines storage locations
for uploaded files and generated outputs, as well as model configurations
for transcription and question generation.

Directory Structure:
    project/
    ├── app/
    │   └── config.py
    ├── uploads/
    │   └── <video_id>/
    └── output/
        └── <video_id>/
            ├── meta.json
            ├── audio.wav
            └── transcript.json

Configuration:
    - WHISPER_MODEL:
        Faster-Whisper model used for speech-to-text conversion.

    - QUESTION_MODEL:
        Hugging Face model used for generating questions from transcript
        content.
"""

# Root directory of the project.
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory for storing uploaded videos.
UPLOAD_DIR = BASE_DIR / "uploads"

# Directory for storing generated artifacts such as:
#   - metadata
#   - extracted audio
#   - transcripts
#   - generated outputs
OUTPUT_DIR = BASE_DIR / "output"


# --------------------------------------------------------------------
# Model Configuration
# --------------------------------------------------------------------

# Faster-Whisper model configuration.
#
# Available options:
#   - tiny   : Fastest, lowest accuracy.
#   - base   : Balanced speed and accuracy (recommended for POC).
#   - small  : Better accuracy, slower inference.
#   - medium : Higher accuracy, increased resource usage.
#   - large  : Best accuracy, highest resource requirements.
WHISPER_MODEL = "base"

# Hugging Face question generation model.
#
# This model generates comprehension-style questions from transcript
# text without requiring any external APIs or API keys.
QUESTION_MODEL = "iarfmoose/t5-base-question-generator"


# --------------------------------------------------------------------
# Directory Initialization
# --------------------------------------------------------------------

# Ensure required directories exist at application startup.
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)