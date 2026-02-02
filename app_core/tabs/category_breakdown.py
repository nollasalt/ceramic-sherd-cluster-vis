from dash import dcc, html


def build_category_breakdown_tab():
    return dcc.Tab(
        label='类别构成',
        value='category-breakdown',
        children=[
            html.Div([
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
