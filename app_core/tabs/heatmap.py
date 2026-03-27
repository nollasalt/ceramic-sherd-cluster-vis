"""聚类特征热力图标签页布局定义。"""

from dash import dcc, html


def build_heatmap_tab():
    """构建热力图标签页容器，用于承载聚类中心特征热图。"""
    return dcc.Tab(
        label='聚类特征热力图',
        value='heatmap',
        children=[
            html.Div([
                # ── 使用说明 ───────────────────────────────────────────────────
                html.Div([
                    html.P('📖 使用说明', style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '8px', 'color': '#2c3e50'}),
                    html.P('本页面展示各簇中心的特征热力图，颜色深浅表示特征值大小。通过对比不同簇在各特征维度上的差异，可以快速识别每个簇的典型特征模式，理解聚类结果的特征空间分布。',
                           style={'fontSize': '12px', 'color': '#555', 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'padding': '12px 14px', 'backgroundColor': '#e0f2f1', 'border': '1px solid #80cbc4',
                         'borderRadius': '8px', 'marginBottom': '12px'}),

                dcc.Loading(
                    type='default',
                    children=html.Div(id='heatmap-container', style={'height': 'calc(100vh - 180px)', 'width': '100%'}),
                )
            ], style={'marginTop': '12px'}),
        ],
    )
