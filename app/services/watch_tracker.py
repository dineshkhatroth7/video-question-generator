import json
from datetime import date
from pathlib import Path

from app.config import OUTPUT_DIR


class WatchTracker:
    """Track per-user daily watch time (POC: single local user)."""

    def __init__(self, store_path: Path | None = None):
        self.store_path = store_path or (OUTPUT_DIR / "watch_log.json")
        self._data = self._load()

    def _load(self) -> dict:
        if self.store_path.exists():
            return json.loads(self.store_path.read_text(encoding="utf-8"))
        return {}

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def _today_key(self) -> str:
        return date.today().isoformat()

    def get_watch_time(self, video_id: str) -> float:
        return float(self._data.get(self._today_key(), {}).get(video_id, 0.0))

    def add_watch_time(self, video_id: str, seconds: float) -> float:
        today = self._today_key()
        self._data.setdefault(today, {})
        current = float(self._data[today].get(video_id, 0.0))
        updated = current + seconds
        self._data[today][video_id] = updated
        self._save()
        return updated

    def set_watch_time(self, video_id: str, seconds: float) -> float:
        today = self._today_key()
        self._data.setdefault(today, {})
        self._data[today][video_id] = max(0.0, seconds)
        self._save()
        return self._data[today][video_id]
