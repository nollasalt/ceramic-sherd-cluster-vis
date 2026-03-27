"""聚类可解释性分析标签页布局定义。"""

from dash import dcc, html

_CARD = {'padding': '14px 16px', 'border': '1px solid #e4e8ef',
         'borderRadius': '10px', 'backgroundColor': '#fff',
         'boxShadow': '0 1px 4px rgba(0,0,0,0.05)', 'marginBottom': '12px'}


def build_interpretability_tab():
    """构建聚类可解释性分析页面。

    展示每个簇的特征重要性、判别特征和簇轮廓。
    """
    return dcc.Tab(
        label='聚类解释',
        value='interpretability',
        children=[
            html.Div([

                # ── 控制区 ─────────────────────────────────────────────────
                html.Div([
                    html.Div([
                        html.Label('选择簇'),
                        dcc.Dropdown(
                            id='interp-cluster-select',
                            placeholder='选择要分析的簇',
                        ),
                    ], style={'flex': '1', 'minWidth': '160px'}),
                ], className='analysis-control-bar'),

                # ── 特征分布可视化 ───────────────────────────────────────────
                html.Div([
                    html.P('特征分布统计', className='dash-card-title'),
                    html.P(
                        '展示该簇内装饰技法、颜色、部位的分布情况，帮助理解聚类依据。',
                        style={'fontSize': '12px', 'color': '#666', 'marginBottom': '8px'}
                    ),
                    dcc.Loading(
                        type='default',
                        children=html.Div([
                            dcc.Graph(id='interp-distribution-charts', style={'height': '350px'}),
                        ]),
                    ),
                ], style=_CARD),

                # ── 簇轮廓摘要 ───────────────────────────────────────────────
                html.Div([
                    html.P('簇轮廓摘要', className='dash-card-title'),
                    html.Div(id='interp-profile'),
                ], style=_CARD),

            ], style={'padding': '14px'}),
        ],
    )
