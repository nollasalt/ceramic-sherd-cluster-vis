"""单层详情标签页布局定义。"""

from dash import dcc, html


def build_single_layer_tab():
    """构建单层详情分析页面。"""
    return dcc.Tab(
        label='单层详情',
        value='single-layer',
        children=[
            html.Div([
                # ── 使用说明 ───────────────────────────────────────────────────
                html.Div([
                    html.P('📖 使用说明', style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '8px', 'color': '#2c3e50'}),
                    html.P('本页面专注于单个地层的深度分析，展示该层内所有簇的分布、代表样本和特征。适合分层聚类后逐层检查聚类质量。',
                           style={'fontSize': '12px', 'color': '#555', 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'padding': '12px 14px', 'backgroundColor': '#e8f5e9', 'border': '1px solid #81c784',
                         'borderRadius': '8px', 'marginBottom': '12px'}),

                # ── 控制栏 ─────────────────────────────────────────────────────
                html.Div([
                    html.Div([
                        html.Label('选择地层:', style={'fontSize': '13px', 'marginRight': '8px', 'fontWeight': '600'}),
                        dcc.Dropdown(
                            id='single-layer-selector',
                            placeholder='选择一个地层',
                            style={'width': '200px'},
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),
                ]),

                # ── 内容区 ─────────────────────────────────────────────────────
                dcc.Loading(
                    id='single-layer-loading',
                    type='default',
                    children=html.Div(id='single-layer-content', style={'marginTop': '12px'}),
                ),
            ], style={'marginTop': '12px', 'padding': '0 8px'}),
        ],
    )
