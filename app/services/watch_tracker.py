import json
from datetime import date
from pathlib import Path

from app.config import OUTPUT_DIR


class WatchTracker:
    """
    Track and persist daily watch time for videos.

    This class provides a lightweight file-based storage mechanism
    for maintaining video watch durations on a per-day basis.
    It is intended for Proof of Concept (POC) or local development
    environments where a database is not required.

    Watch times are stored in the following JSON structure:

    {
        "2026-07-15": {
            "video_1": 120.5,
            "video_2": 450.0
        },
        "2026-07-16": {
            "video_1": 90.0
        }
    }

    Attributes:
        store_path (Path):
            Location of the JSON file used to persist watch time data.

        _data (dict):
            In-memory representation of the watch log.
    """

    def __init__(self, store_path: Path | None = None):
        """
        Initialize the WatchTracker.

        Args:
            store_path (Path | None, optional):
                Custom path for the watch log JSON file. If omitted,
                the default path is `<OUTPUT_DIR>/watch_log.json`.
        """
        self.store_path = store_path or (
            OUTPUT_DIR / "watch_log.json"
        )
        self._data = self._load()

    def _load(self) -> dict:
        """
        Load watch time data from disk.

        Returns:
            dict:
                Parsed JSON data containing watch times. Returns an
                empty dictionary if the file does not exist.
        """
        if self.store_path.exists():
            return json.loads(
                self.store_path.read_text(encoding="utf-8")
            )

        return {}

    def _save(self) -> None:
        """
        Persist the current watch time data to disk.

        The parent directory is created automatically if it does
        not already exist.
        """
        self.store_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.store_path.write_text(
            json.dumps(self._data, indent=2),
            encoding="utf-8",
        )

    def _today_key(self) -> str:
        """
        Generate today's date key.

        Returns:
            str:
                Current date in ISO 8601 format (YYYY-MM-DD).

        Example:
            >>> tracker._today_key()
            '2026-07-15'
        """
        return date.today().isoformat()

    def get_watch_time(self, video_id: str) -> float:
        """
        Retrieve the watch time for a video for the current day.

        Args:
            video_id (str):
                Unique identifier of the video.

        Returns:
            float:
                Number of seconds watched today. Returns 0.0 if no
                record exists.
        """
        return float(
            self._data
            .get(self._today_key(), {})
            .get(video_id, 0.0)
        )

    def add_watch_time(
        self,
        video_id: str,
        seconds: float,
    ) -> float:
        """
        Add watch time to an existing video entry for today.

        If the video has not been watched previously today, a new
        entry is created automatically.

        Args:
            video_id (str):
                Unique identifier of the video.

            seconds (float):
                Additional number of seconds watched.

        Returns:
            float:
                Updated cumulative watch time for the video.

        Example:
            >>> tracker.add_watch_time("abc123", 120)
            120.0
            >>> tracker.add_watch_time("abc123", 30)
            150.0
        """
        today = self._today_key()

        self._data.setdefault(today, {})

        current = float(
            self._data[today].get(video_id, 0.0)
        )

        updated = current + seconds

        self._data[today][video_id] = updated

        self._save()

        return updated

    def set_watch_time(
        self,
        video_id: str,
        seconds: float,
    ) -> float:
        """
        Explicitly set the watch time for a video.

        Negative values are automatically converted to 0.0.

        Args:
            video_id (str):
                Unique identifier of the video.

            seconds (float):
                Desired watch duration in seconds.

        Returns:
            float:
                Final watch duration stored for the video.

        Example:
            >>> tracker.set_watch_time("abc123", 300)
            300.0
        """
        today = self._today_key()

        self._data.setdefault(today, {})

        self._data[today][video_id] = max(
            0.0,
            seconds,
        )

        self._save()

        return self._data[today][video_id]