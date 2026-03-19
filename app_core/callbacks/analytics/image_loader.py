"""大图加载回调（根据路径返回高分辨率 base64 图像）。"""
from pathlib import Path

import dash
from dash import Input, Output

from data_processing import img_to_base64_full


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
        """加载并返回原图的高分辨率 base64 数据。"""
        if not image_path or image_path == '':
            return dash.no_update
        try:
            full_path = resolve_full_path(image_path)
            if full_path and full_path.exists():
                full_res_image = img_to_base64_full(str(full_path))
                return full_res_image
            return dash.no_update
        except Exception:
            return dash.no_update
