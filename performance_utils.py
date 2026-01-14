"""
性能优化工具模块
包含缓存、性能监控和优化函数
"""

import time
import functools
from typing import Dict, Any, Optional
import pandas as pd


class PerformanceCache:
    """简单的性能缓存类"""
    
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存项"""
        # 如果缓存已满，删除最旧的项
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), 
                           key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()


# 全局缓存实例
plot_cache = PerformanceCache(max_size=50)
image_cache = PerformanceCache(max_size=200)


def timing_decorator(func):
    """性能计时装饰器（静默模式）"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper


def cache_plot_result(func):
    """缓存图表结果的装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 生成缓存键
        cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
        
        # 尝试从缓存获取
        cached_result = plot_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 计算新结果并缓存
        result = func(*args, **kwargs)
        plot_cache.set(cache_key, result)
        return result
    return wrapper


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """优化DataFrame的内存使用"""
    original_memory = df.memory_usage(deep=True).sum()
    
    # 优化数值列的数据类型
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # 优化对象列
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < len(df) * 0.5:  # 如果唯一值少于50%，转换为category
            df[col] = df[col].astype('category')
    
    new_memory = df.memory_usage(deep=True).sum()
    return df


def batch_process_images(image_paths: list, batch_size: int = 10):
    """批处理图片加载以提高性能"""
    from src.data.processing import img_to_base64
    
    results = []
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batch_results = []
        
        pass  # 静默处理
        
        for img_path in batch:
            # 检查缓存
            cache_key = f"img_{img_path}"
            cached_img = image_cache.get(cache_key)
            
            if cached_img is not None:
                batch_results.append(cached_img)
            else:
                try:
                    img_b64 = img_to_base64(img_path, max_size=80)
                    image_cache.set(cache_key, img_b64)
                    batch_results.append(img_b64)
                except Exception:
                    batch_results.append(None)
        
        results.extend(batch_results)
    
    return results


def get_memory_usage():
    """获取当前内存使用情况"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }


def log_memory_usage(func_name: str):
    """记录内存使用情况（静默模式）"""
    pass