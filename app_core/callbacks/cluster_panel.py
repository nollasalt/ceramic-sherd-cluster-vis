#负责“簇图片面板”的回调逻辑
from pathlib import Path

import dash
from dash import Input, Output, State, html

from data_processing import img_to_base64

PAGE_SIZE = 20


def register_cluster_panel_callbacks(app):
    """Register cluster image panel pagination callbacks."""

    @app.callback(
        Output('cluster-panel', 'children'),
        Output('page-indicator', 'children'),
        Input('cluster-images-store', 'data'),
        Input('cluster-page', 'data')
    )
    def update_cluster_panel(image_paths, page):
        if not image_paths:
            return html.Div('点击点以加载该簇的图片'), ''
        total = len(image_paths)
        max_page = (total + PAGE_SIZE - 1) // PAGE_SIZE
        page = max(1, min(page or 1, max_page))
        start = (page - 1) * PAGE_SIZE
        end = min(start + PAGE_SIZE, total)
        thumbs = []
        for i, pth in enumerate(image_paths[start:end]):
            b64 = img_to_base64(Path(pth), max_size=180)
            if b64:
                thumbs.append(html.Img(
                    src=b64,
                    id=f'cluster-img-{start + i}',
                    **{'data-image-path': Path(pth).name},
                    style={
                        'height': '120px',
                        'margin': '4px',
                        'border': '1px solid #ccc',
                        'cursor': 'pointer'
                    },
                    title='点击放大查看'
                ))
            else:
                thumbs.append(html.Div(str(Path(pth).name)))
        grid = html.Div(thumbs, style={'display': 'flex', 'flexWrap': 'wrap'})
        indicator = f'Page {page} / {max_page} ({total} images)'
        return grid, indicator

    @app.callback(
        Output('cluster-page', 'data'),
        Input('cluster-prev', 'n_clicks'),
        Input('cluster-next', 'n_clicks'),
        State('cluster-page', 'data'),
        State('cluster-images-store', 'data')
    )
    def change_page(prev_clicks, next_clicks, current_page, image_paths):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_page or 1
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        total = len(image_paths) if image_paths else 0
        max_page = (total + PAGE_SIZE - 1) // PAGE_SIZE if total > 0 else 1
        current = current_page or 1
        if triggered == 'cluster-prev':
            new = max(1, current - 1)
        elif triggered == 'cluster-next':
            new = min(max_page, current + 1)
        else:
            new = current
        return new

    return app
