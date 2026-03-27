"""类别构成标签页布局定义。"""

from dash import dcc, html


def build_category_breakdown_tab():
    """构建类别构成页面，用于按簇/单位查看类别分布。"""
    return dcc.Tab(
        label='类别构成',
        value='category-breakdown',
        children=[
            html.Div([
                # ── 使用说明 ───────────────────────────────────────────────────
                html.Div([
                    html.P('📖 使用说明', style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '8px', 'color': '#2c3e50'}),
                    html.P('本页面展示各簇或各单位中不同类别（部位、器类、地层）的构成比例。通过堆叠柱状图直观显示每个分组的类别分布，帮助理解簇的组成特征或单位的器物组合模式。',
                           style={'fontSize': '12px', 'color': '#555', 'lineHeight': '1.6', 'margin': '0'}),
                ], style={'padding': '12px 14px', 'backgroundColor': '#f8bbd0', 'border': '1px solid #f48fb1',
                         'borderRadius': '8px', 'marginBottom': '12px'}),

                html.Div([
                    html.Label('类别字段:'),
                    dcc.Dropdown(
                        id='category-field-selector',
                        options=[
                            {'label': '部位 (part_C)', 'value': 'part_C'},
                            {'label': '类型 (type_C)', 'value': 'type_C'},
                            {'label': '单位 (unit_C)', 'value': 'unit_C'},
                        ],
                        value='part_C',
                        clearable=False,
                        style={'width': '220px'},
                    ),
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Label('横轴'),
                    dcc.RadioItems(
                        id='category-x-axis',
                        options=[
                            {'label': '按簇', 'value': 'cluster'},
                            {'label': '按单位 (unit_C)', 'value': 'unit_C'},
                        ],
                        value='cluster',
                        labelStyle={'marginRight': '12px'},
                    ),
                ], style={'marginBottom': '8px'}),
                dcc.Loading(
                    type='default',
                    children=dcc.Graph(id='category-breakdown-graph', style={'height': 'calc(100vh - 230px)'}),
                ),
            ], style={'marginTop': '12px', 'padding': '0 8px'}),
        ],
    )
