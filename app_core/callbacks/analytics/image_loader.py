"""大图加载回调（模态框高分辨率图像）+ Flask 图像直链路由（缩略图）。"""
from pathlib import Path
from urllib.parse import unquote

import dash
from dash import Input, Output

from data_processing import img_to_base64_full


def register_image_server(app, *, search_dirs):
    """注册 Flask 路由 /img/<filename>，让浏览器直接请求图像文件。

    缩略图通过 <img src='/img/filename' loading='lazy'> 加载，
    回调 JSON 响应不再携带 base64 数据，大幅降低传输量。
    """
    from flask import send_file, abort

    @app.server.route('/img/<filename>')
    def serve_image_direct(filename):
        safe_name = Path(unquote(filename)).name
        if not safe_name or '.' not in safe_name:
            abort(404)
        for base in search_dirs:
            path = Path(base) / safe_name
            if path.exists():
                return send_file(str(path.resolve()), max_age=3600)
        abort(404)


def register_image_loader_callbacks(app, *, search_dirs):
    def resolve_full_path(image_path: str) -> Path | None:
        """在配置的图像目录中解析大图文件路径。"""
        if not image_path:
            return None
        target = Path(image_path)
        candidates = [target]
        if not target.is_absolute():
            candidates.append(Path(target.name))

        for base in search_dirs:
            base = Path(base)
            if not base.exists():
                continue
            for cand in candidates:
                cand_path = base / cand
                if cand_path.exists():
                    return cand_path
            try:
                match = next(base.rglob(target.name))
                if match.exists():
                    return match
            except StopIteration:
                pass
        return None

    @app.callback(
        Output('modal-image', 'src'),
        [Input('image-path-input', 'value')],
        prevent_initial_call=True
    )
    def load_full_image(image_path):
        """加载并返回原图的高分辨率 base64 数据（仅模态框点击时触发）。"""
        if not image_path or image_path == '':
            return dash.no_update
        try:
            full_path = resolve_full_path(image_path)
            if full_path and full_path.exists():
                return img_to_base64_full(str(full_path))
            return dash.no_update
        except Exception:
            return dash.no_update
