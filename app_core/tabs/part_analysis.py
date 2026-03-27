"""器部分布分析标签页布局（part_C × cluster）。"""

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


def build_part_analysis_tab():
    """构建器部分布分析页面。

    器部（part_C）× 簇混淆热力图：
    - 某簇全为口沿 → 捕获器形特征
    - 某簇混有各器部 → 可能捕获纹饰或胎土特征
    """
    return dcc.Tab(
        label='器部分布',
        value='part-analysis',
        children=[
            html.Div([
                # ── 使用说明 ───────────────────────────────────────────────────
                html.Div([
                    html.P('📖 使用说明', style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '8px', 'color': '#2c3e50'}),
                    html.P('本页面展示器部（口沿、腹部等）在各簇中的分布。若某簇全为同一器部，说明聚类捕获了器形特征；若某簇混有多种器部，则可能捕获了纹饰或胎土特征。器部熵指标量化了簇的器部多样性，帮助理解聚类依据的特征类型。',
                           style={'fontSize': '12px', 'color': '#555', 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'padding': '12px 14px', 'backgroundColor': '#eceff1', 'border': '1px solid #b0bec5',
                         'borderRadius': '8px', 'marginBottom': '12px'}),

                # ── 控制区 ─────────────────────────────────────────────────
                html.Div([
                    html.Div([
                        html.Label('归一化方式'),
                        dcc.RadioItems(
                            id='part-val-norm',
                            options=[
                                {'label': '绝对数', 'value': 'count'},
                                {'label': '按器部（行）', 'value': 'by_part'},
                                {'label': '按簇（列）', 'value': 'by_cluster'},
                            ],
                            value='by_cluster',
                            labelStyle={'display': 'block', 'marginBottom': '3px'},
                        ),
                    ], style={'minWidth': '140px'}),

                    html.Div([
                        html.Label('最少样本数（过滤小簇）'),
                        dcc.Slider(
                            id='part-val-min-samples',
                            min=1, max=50, step=1, value=5,
                            marks={1: '1', 10: '10', 20: '20', 30: '30', 50: '50'},
                            tooltip={'placement': 'bottom', 'always_visible': False},
                        ),
                    ], style={'flex': '3', 'minWidth': '220px'}),
                ], className='analysis-control-bar'),

                # ── 混淆矩阵热力图 ──────────────────────────────────────────
                html.Div([
                    html.P('器部 × 簇 分布矩阵', style=_CARD_TITLE),
                    html.P(
                        '行 = 器部（part_C），列 = 簇编号。'
                        '按簇归一化时，颜色深浅反映该簇内各器部的占比，'
                        '可判断簇"捕获"的是器形还是纹饰/胎土特征。',
                        style={'fontSize': '11px', 'color': '#888', 'margin': '-4px 0 8px'},
                    ),
                    dcc.Loading(
                        type='default',
                        children=dcc.Graph(id='part-val-heatmap', style={'height': 'calc(100vh - 360px)'}),
                    ),
                ], style=_CARD),

                # ── 簇语义标签 + 器部熵 ──────────────────────────────────────
                html.Div([
                    html.Div([
                        html.Div(id='part-val-metrics'),
                    ], style={
                        **_CARD, 'flex': '1', 'minWidth': '260px',
                        'marginBottom': '0', 'backgroundColor': '#f8fafc',
                    }),

                    html.Div([
                        html.P('各簇器部构成明细', style=_CARD_TITLE),
                        html.Div(id='part-val-detail'),
                    ], style={**_CARD, 'flex': '2', 'minWidth': '320px', 'marginBottom': '0'}),

                ], style={
                    'display': 'flex', 'gap': '12px',
                    'flexWrap': 'wrap', 'alignItems': 'flex-start',
                }),

            ], style={'padding': '14px'}),
        ],
    )
