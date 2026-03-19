"""器类验证矩阵标签页布局定义。"""

from dash import dcc, html

_CARD = {
    'padding': '14px 16px',
    'border': '1px solid #e4e8ef',
    'borderRadius': '10px',
    'backgroundColor': '#fff',
    'boxShadow': '0 1px 4px rgba(0,0,0,0.05)',
    'marginBottom': '12px',
}
_CARD_TITLE = {
    'fontSize': '13px', 'fontWeight': '700', 'color': '#2c3e50',
    'margin': '0 0 10px 0', 'letterSpacing': '0.02em',
}


def build_type_validation_tab():
    """构建器类验证矩阵页面。

    包含：type_C × cluster_id 混淆热力图、ARI/NMI 纯度指数、逐器类纯度明细表。
    """
    return dcc.Tab(
        label='器类验证矩阵',
        value='type-validation',
        children=[
            html.Div([

                # ── 控制区 ─────────────────────────────────────────────────
                html.Div([
                    html.Div([
                        html.Label('归一化方式'),
                        dcc.RadioItems(
                            id='type-val-norm',
                            options=[
                                {'label': '绝对数', 'value': 'count'},
                                {'label': '按器类（行）', 'value': 'by_type'},
                                {'label': '按簇（列）', 'value': 'by_cluster'},
                            ],
                            value='by_type',
                            labelStyle={'display': 'block', 'marginBottom': '3px'},
                        ),
                    ], style={'minWidth': '140px'}),

                    html.Div([
                        html.Label('显示前 N 器类'),
                        dcc.Slider(
                            id='type-val-topn',
                            min=5, max=30, step=1, value=15,
                            marks={5: '5', 10: '10', 15: '15', 20: '20', 30: '30'},
                            tooltip={'placement': 'bottom', 'always_visible': False},
                        ),
                    ], style={'flex': '3', 'minWidth': '220px'}),
                ], className='analysis-control-bar'),

                # ── 混淆矩阵热力图 ──────────────────────────────────────────
                html.Div([
                    html.P('器类 × 簇 混淆矩阵', style=_CARD_TITLE),
                    html.P(
                        '行 = 器类（type_C 人工标注），列 = 簇编号（算法分组）。'
                        '颜色越深表示该器类样本越集中于该簇。',
                        style={'fontSize': '11px', 'color': '#888', 'margin': '-4px 0 8px'},
                    ),
                    dcc.Loading(
                        type='default',
                        children=dcc.Graph(id='type-val-heatmap', style={'height': '500px'}),
                    ),
                ], style=_CARD),

                # ── 纯度指数 + 逐器类明细 ────────────────────────────────────
                html.Div([
                    html.Div([
                        html.Div(id='type-val-metrics'),
                    ], style={
                        **_CARD, 'flex': '1', 'minWidth': '260px',
                        'marginBottom': '0', 'backgroundColor': '#f8fafc',
                    }),

                    html.Div([
                        html.P('逐器类纯度明细', style=_CARD_TITLE),
                        html.Div(id='type-val-detail'),
                    ], style={**_CARD, 'flex': '2', 'minWidth': '320px', 'marginBottom': '0'}),

                ], style={
                    'display': 'flex', 'gap': '12px',
                    'flexWrap': 'wrap', 'alignItems': 'flex-start',
                }),

            ], style={'padding': '14px'}),
        ],
    )
