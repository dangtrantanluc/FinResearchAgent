"""Read KEY=VALUE lines from .env into the environment, without overriding what is already set."""
import logging
import os
from pathlib import Path


def load_env(root: str | Path | None = None) -> None:
    path = Path(root or Path(__file__).resolve().parents[1]) / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class _WatcherNoise(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Examining the path of" not in record.getMessage()


def silence_source_watcher() -> None:
    """Drop Streamlit's "Examining the path of <module> raised" warnings.

    Its hot-reload watcher inspects every imported module. With transformers loaded that touches
    about a hundred lazily imported vision modules, each failing on the missing torchvision and
    logging a full traceback. A filter is used because Streamlit resets logger levels itself.
    """
    logger = logging.getLogger("streamlit.watcher.local_sources_watcher")
    if not any(isinstance(f, _WatcherNoise) for f in logger.filters):
        logger.addFilter(_WatcherNoise())
