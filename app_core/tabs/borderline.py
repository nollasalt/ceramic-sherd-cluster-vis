"""边界样本标签页布局定义。"""

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
    'margin': '0 0 8px 0', 'letterSpacing': '0.02em',
}


def build_borderline_tab():
    """构建边界样本展示标签页。

    展示每个簇中距离本簇中心最远、同时距离其他簇中心最近的样本——
    这些样本是拼对候选、分类错误或独特器物的首选审查对象。
    """
    return dcc.Tab(
        label='边界样本',
        value='borderline',
        children=[
            html.Div([

                # ── 控制区 ─────────────────────────────────────────────────
                html.Div([
                    html.Div([
                        html.Label('每簇显示数量'),
                        dcc.Slider(
                            id='borderline-per-cluster',
                            min=1, max=8, step=1, value=3,
                            marks={1: '1', 2: '2', 3: '3', 5: '5', 8: '8'},
                            tooltip={'placement': 'bottom', 'always_visible': False},
                        ),
                    ], style={'flex': '2', 'minWidth': '200px'}),

                    html.Div([
                        html.Label('边界度阈值（仅显示 ≥ 此分数的样本）'),
                        dcc.Slider(
                            id='borderline-threshold',
                            min=0.0, max=1.0, step=0.05, value=0.5,
                            marks={0: '0', 0.5: '0.5', 0.75: '0.75', 1.0: '1.0'},
                            tooltip={'placement': 'bottom', 'always_visible': False},
                        ),
                    ], style={'flex': '3', 'minWidth': '220px'}),
                ], className='analysis-control-bar'),

                # ── 说明提示 ────────────────────────────────────────────────
                html.Div([
                    html.Span('边界度 = ', style={'fontWeight': '600', 'color': '#555'}),
                    html.Span('本簇内距离 ÷ 最近其他簇距离', style={'fontFamily': 'monospace', 'color': '#2c6fad'}),
                    html.Span('。分数越接近 1（或超过 1），样本越靠近另一个簇的边界。'
                              '这些样本是', style={'color': '#666'}),
                    html.Span(' 拼对候选 / 分类错误 / 独特器物 ', style={
                        'color': '#e67e22', 'fontWeight': '600',
                    }),
                    html.Span('的首选排查对象。', style={'color': '#666'}),
                ], style={
                    'fontSize': '12px', 'padding': '8px 12px',
                    'backgroundColor': '#fffbf0', 'border': '1px solid #f0e0b0',
                    'borderRadius': '8px', 'marginBottom': '12px', 'lineHeight': '1.7',
                }),

                # ── 边界样本图格（分簇展示） ─────────────────────────────────
                dcc.Loading(
                    id='borderline-loading',
                    type='default',
                    children=html.Div(
                        id='borderline-grid',
                        style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '12px'},
                    ),
                ),

                # ── 分页控件 ─────────────────────────────────────────────────
                dcc.Store(id='borderline-page-index', data=1),
                html.Div([
                    html.Button(
                        '上一页', id='borderline-page-prev', n_clicks=0,
                        style={
                            'backgroundColor': '#fff', 'color': '#333',
                            'border': '1px solid #ccc', 'padding': '6px 14px',
                            'cursor': 'pointer', 'borderRadius': '4px',
                        },
                    ),
                    html.Div(id='borderline-page-status',
                             style={'fontSize': '13px', 'color': '#555'}),
                    html.Button(
                        '下一页', id='borderline-page-next', n_clicks=0,
                        style={
                            'backgroundColor': '#fff', 'color': '#333',
                            'border': '1px solid #ccc', 'padding': '6px 14px',
                            'cursor': 'pointer', 'borderRadius': '4px',
                        },
                    ),
                ], style={
                    'display': 'flex', 'justifyContent': 'space-between',
                    'alignItems': 'center', 'marginTop': '8px', 'marginBottom': '16px',
                }),

                # ── 模糊边界对统计 ───────────────────────────────────────────
                html.Div([
                    html.Div([
                        html.P('模糊边界对', style=_CARD_TITLE),
                        html.Span('各簇间边界模糊的样本数（越多说明两簇越容易混淆，可考虑合并或重新审查）',
                                  style={'fontSize': '11px', 'color': '#888'}),
                    ], style={'marginBottom': '10px'}),
                    html.Div(id='borderline-pair-stats'),
                ], style={**_CARD, 'backgroundColor': '#f8fafc'}),

            ], style={'padding': '14px'}),
        ],
    )
