"""Simple in-memory cache for shared dataset/state across callbacks."""

from typing import Any, Dict


_DATA_CACHE: Dict[str, Any] | None = None


def set_data_cache(data: Dict[str, Any]) -> None:
    # Store the shared dataset/metadata so callbacks can reuse it without sending big payloads to clients
    global _DATA_CACHE
    _DATA_CACHE = data


def get_data_cache() -> Dict[str, Any]:
    # Retrieve cached data; raise early if the cache was not initialized
    if _DATA_CACHE is None:
        raise RuntimeError("Data cache not initialized")
    return _DATA_CACHE
