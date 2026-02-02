#管理“对比面板”的回调
from pathlib import Path
import json

import dash
from dash import Input, Output, State, html

from data_processing import img_to_base64


def register_compare_callbacks(app):
    """Register compare panel related callbacks."""

    @app.callback(
        Output('compare-selected-store', 'data'),
        Input('compare-add', 'n_clicks'),
        Input('compare-clear', 'n_clicks'),
        Input('compare-clear-bottom', 'n_clicks'),
        Input({'type': 'compare-remove', 'index': dash.dependencies.ALL}, 'n_clicks'),
        State('compare-selected-store', 'data'),
        State('last-selected-store', 'data')
    )
    def update_compare_store(add_clicks, clear_clicks, clear_clicks_bottom, remove_clicks, selected_items, last_selected):
        selected_items = selected_items or []
        ctx = dash.callback_context
        if not ctx.triggered:
            return selected_items
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered in ('compare-clear', 'compare-clear-bottom'):
            return []
        if triggered.startswith('{'):
            try:
                info = json.loads(triggered)
            except ValueError:
                info = {}
            if info.get('type') == 'compare-remove':
                target_id = str(info.get('index', ''))
                return [c for c in selected_items if str(c.get('sample_id')) != target_id]
        if triggered == 'compare-add':
            if not last_selected or not last_selected.get('sample_id'):
                return selected_items
            sid = str(last_selected.get('sample_id'))
            filtered = [c for c in selected_items if c.get('sample_id') != sid]
            filtered.append(last_selected)
            return filtered
        return selected_items

    @app.callback(
        Output('compare-panel', 'children'),
        Input('compare-selected-store', 'data'),
        Input('compare-size', 'value'),
        Input('compare-layout', 'value')
    )
    def render_compare(selected_items, card_size, layout_mode):
        if not selected_items:
            return html.Div('点击散点图选中样本后，按“添加到比较”即可在此并排查看。', style={'color': '#666'})

        size = card_size or 220
        img_h = max(120, min(360, size))
        card_w = img_h + 40

        cards = []
        for item in selected_items:
            pth = item.get('path')
            b64 = img_to_base64(Path(pth), max_size=img_h) if pth else None
            cards.append(html.Div([
                html.Div(f"Cluster {item.get('cluster', '')}", style={'fontSize': '12px', 'color': '#666'}),
                html.Img(
                    src=b64 if b64 else '',
                    style={'height': f'{img_h}px', 'border': '1px solid #ccc', 'borderRadius': '4px', 'backgroundColor': '#fafafa'},
                    **({'data-image-path': Path(pth).name} if pth else {})
                ),
                html.Div(item.get('name', '未知'), style={'marginTop': '6px', 'fontSize': '13px', 'fontWeight': '500'}),
                html.Button(
                    '移除',
                    id={'type': 'compare-remove', 'index': str(item.get('sample_id', ''))},
                    n_clicks=0,
                    style={
                        'marginTop': '8px',
                        'padding': '4px 10px',
                        'border': '1px solid #ccc',
                        'borderRadius': '4px',
                        'backgroundColor': '#f8f8f8',
                        'cursor': 'pointer'
                    }
                )
            ], style={'width': f'{card_w}px', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'gap': '4px'}))

        container_style = {
            'display': 'flex',
            'gap': '16px',
            'padding': '4px'
        }
        if layout_mode == 'row':
            container_style.update({'flexWrap': 'nowrap', 'overflowX': 'auto'})
        else:
            container_style.update({'flexWrap': 'wrap'})

        return html.Div(cards, style=container_style)

    return app
