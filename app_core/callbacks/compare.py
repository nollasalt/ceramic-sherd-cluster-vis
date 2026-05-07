"""样本对比面板回调。"""



#管理"对比面板"的回调

from pathlib import Path

import json



import dash

from dash import Input, Output, State, html





def register_compare_callbacks(app):

    """注册手动对比面板相关回调。"""



    @app.callback(

        Output('visualization-tabs', 'value'),

        Input('visualization-tabs', 'value'),

        prevent_initial_call=True,

    )

    def redirect_group_tabs(tab_value):

        """防止分组标题伪标签页被选中（CSS pointer-events 是主防线，此为兜底）。"""

        _REDIRECTS = {

            'group-overview': 'representatives',

            'group-quality': 'cluster-size',

            'group-analysis': 'category-breakdown',

            'group-stratigraphy': 'stratigraphy',

        }

        if tab_value in _REDIRECTS:

            return _REDIRECTS[tab_value]

        return dash.no_update



    @app.callback(

        Output('compare-section', 'style'),

        Input('visualization-tabs', 'value'),

    )

    def toggle_compare_section(tab_value):

        """仅在散点图标签页时显示比较视图。"""

        base = {'borderTop': '1px solid #eee', 'paddingTop': '8px', 'marginTop': '8px'}

        if tab_value == 'scatter':

            return base

        return {**base, 'display': 'none'}



    @app.callback(

        Output('sample-panel', 'style'),

        Input('visualization-tabs', 'value'),

    )

    def toggle_sample_panel(tab_value):

        """仅在散点图标签页时显示样本面板。"""

        base = {'marginTop': '12px', 'minHeight': '220px', 'borderTop': '1px solid #ddd', 'paddingTop': '8px'}

        if tab_value == 'scatter':

            return base

        return {**base, 'display': 'none'}



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

        """维护对比列表状态（添加、移除、清空）。"""

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

        """根据当前选中样本渲染对比卡片区域。"""

        if not selected_items:

            return html.Div('点击散点图选中样本后，按"添加到比较"即可在此并排查看。', style={'color': '#666'})



        size = card_size or 220

        img_h = max(120, min(360, size))

        card_w = img_h + 40



        cards = []

        for item in selected_items:

            pth = item.get('path')

            img_src = f'/img/{Path(pth).name}' if pth else None

            cards.append(html.Div([

                html.Div(f"Cluster {item.get('cluster', '')}", style={'fontSize': '12px', 'color': '#666'}),

                html.Img(

                    src=img_src or '',


                    style={'height': f'{img_h}px', 'border': '1px solid #ccc', 'borderRadius': '4px', 'backgroundColor': '#fafafa'},

                    **({'data-image-path': Path(pth).name} if pth else {}),
                    title='点击放大查看' if pth else '图片不可用',

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

