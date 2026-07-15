import subprocess
from pathlib import Path

import imageio_ffmpeg


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """
    Extract audio from a video file and save it as a WAV file.

    This function uses the FFmpeg executable bundled with the
    `imageio-ffmpeg` package, eliminating the need for a separate
    FFmpeg installation on the system. The extracted audio is
    converted to a mono (single-channel) WAV file with a sampling
    rate of 16 kHz, which is suitable for speech-to-text models
    such as Whisper and Faster-Whisper.

    Args:
        video_path (Path):
            Path to the input video file (e.g., MP4, AVI, MOV).

        output_path (Path):
            Destination path for the extracted audio file. The
            parent directory will be created automatically if it
            does not already exist.

    Returns:
        Path:
            The path to the generated audio file.

    Raises:
        RuntimeError:
            Raised if FFmpeg fails to extract the audio. The error
            message from FFmpeg is included in the exception.

    Example:
        >>> from pathlib import Path
        >>> extract_audio(
        ...     Path("lecture.mp4"),
        ...     Path("data/audio/lecture.wav")
        ... )
        PosixPath('data/audio/lecture.wav')
    """
    # Get the bundled FFmpeg executable.
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    # Ensure the output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg command to extract audio:
    # -vn          : Disable video recording.
    # -acodec      : Set audio codec to PCM 16-bit little-endian.
    # -ar          : Set sample rate to 16 kHz.
    # -ac          : Convert to mono audio.
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]

    # Execute the FFmpeg command.
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Raise an exception if audio extraction fails.
    if result.returncode != 0:
        raise RuntimeError(
            f"Audio extraction failed: {result.stderr.strip()}"
        )

    return output_path