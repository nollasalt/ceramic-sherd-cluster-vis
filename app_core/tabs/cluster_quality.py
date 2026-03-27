"""聚类质量标签页布局定义。"""

from dash import dcc, html


def build_cluster_quality_tab():
    """构建聚类质量页面，展示质量指标卡、风险条图与细节表。"""
    return dcc.Tab(
        label='聚类质量',
        value='cluster-quality',
        children=[
            html.Div([
                # ── 使用说明 ───────────────────────────────────────────────────
                html.Div([
                    html.P('📖 使用说明', style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '8px', 'color': '#2c3e50'}),
                    html.P('本页面评估聚类整体质量，展示轮廓系数、Davies-Bouldin指数等关键指标。顶部卡片显示全局质量得分，柱状图展示各簇的质量风险，帮助识别需要优化的簇。',
                           style={'fontSize': '12px', 'color': '#555', 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'padding': '12px 14px', 'backgroundColor': '#fff3e0', 'border': '1px solid #ffb74d',
                         'borderRadius': '8px', 'marginBottom': '12px'}),

                dcc.Loading(
                    type='default',
                    children=html.Div([
                        html.Div(id='cluster-quality-cards', style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '12px'}),
                        dcc.Graph(id='cluster-quality-bars', style={'height': '380px', 'width': '100%', 'marginBottom': '8px'}),
                        html.Div(id='cluster-quality-detail', style={'fontSize': '13px', 'color': '#333', 'padding': '0 4px'}),
                    ]),
                ),
                html.Div('颜色指示: 绿=清晰，黄=需关注，红=混杂/易粘连', style={'fontSize': '12px', 'color': '#666', 'marginTop': '4px', 'padding': '0 4px'}),
            ], style={'marginTop': '12px', 'padding': '0 8px'}),
        ],
    )
