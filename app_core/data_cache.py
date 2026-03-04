"""
Simple in-memory cache for shared dataset/state across callbacks.
用于在服务端缓存 DataFrame 和元数据，避免前端传输大对象。
"""

from typing import Any, Dict


_DATA_CACHE: Dict[str, Any] | None = None


def set_data_cache(data: Dict[str, Any]) -> None:
    """写入全局数据缓存，供多个回调共享使用。"""
    # 避免把大体量 DataFrame 反复传到前端，统一在服务端缓存
    global _DATA_CACHE
    _DATA_CACHE = data


def get_data_cache() -> Dict[str, Any]:
    """读取全局数据缓存；若未初始化则抛出异常。"""
    # 统一在这里做初始化检查，避免回调中出现隐式空值错误
    if _DATA_CACHE is None:
        raise RuntimeError("Data cache not initialized")
    return _DATA_CACHE
