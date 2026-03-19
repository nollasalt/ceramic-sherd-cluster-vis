"""使用说明标签页布局定义（纯静态，无回调）。"""

from dash import html


# ── 样式常量 ──────────────────────────────────────────────────────────────────

_CARD = {
    'padding': '16px 20px',
    'border': '1px solid #e4e4e4',
    'borderRadius': '8px',
    'backgroundColor': '#ffffff',
    'marginBottom': '14px',
}

_CARD_ACCENT = {**_CARD, 'borderLeft': '4px solid #2c6fad'}
_CARD_WARN = {**_CARD, 'borderLeft': '4px solid #e8a838'}
_CARD_GREEN = {**_CARD, 'borderLeft': '4px solid #2ca05a'}

_H2 = {'fontSize': '16px', 'fontWeight': '700', 'color': '#1a1a2e', 'marginBottom': '10px', 'marginTop': '0'}
_H3 = {'fontSize': '14px', 'fontWeight': '700', 'color': '#2c6fad', 'marginBottom': '6px', 'marginTop': '12px'}
_P = {'fontSize': '13px', 'color': '#333', 'lineHeight': '1.7', 'marginBottom': '6px', 'marginTop': '0'}
_TAG = {
    'display': 'inline-block',
    'padding': '1px 8px',
    'borderRadius': '10px',
    'fontSize': '12px',
    'fontWeight': '600',
    'marginRight': '6px',
    'marginBottom': '4px',
}
_STEP = {
    'display': 'flex',
    'alignItems': 'flex-start',
    'gap': '12px',
    'marginBottom': '10px',
}
_STEP_NUM = {
    'minWidth': '24px',
    'height': '24px',
    'borderRadius': '50%',
    'backgroundColor': '#2c6fad',
    'color': '#fff',
    'display': 'flex',
    'alignItems': 'center',
    'justifyContent': 'center',
    'fontSize': '12px',
    'fontWeight': '700',
    'flexShrink': '0',
}
_TH = {'padding': '6px 12px', 'backgroundColor': '#f0f4fa', 'fontSize': '12px',
        'fontWeight': '700', 'color': '#333', 'textAlign': 'left', 'borderBottom': '2px solid #d0d8e8'}
_TD = {'padding': '6px 12px', 'fontSize': '12px', 'color': '#333',
        'borderBottom': '1px solid #eee', 'verticalAlign': 'top'}


def _tag(text, color='#2c6fad', bg='#e8f0fb'):
    return html.Span(text, style={**_TAG, 'color': color, 'backgroundColor': bg})


def _step(n, title, body):
    return html.Div([
        html.Div(str(n), style=_STEP_NUM),
        html.Div([
            html.Div(title, style={'fontWeight': '600', 'fontSize': '13px', 'color': '#222', 'marginBottom': '2px'}),
            html.Div(body, style=_P),
        ]),
    ], style=_STEP)


def _section(title, children, style=None):
    return html.Div([
        html.H2(title, style=_H2),
        *children,
    ], style={**(style or _CARD)})


# ── 主构建函数 ────────────────────────────────────────────────────────────────

def build_help_tab():
    """构建静态使用说明标签页，无需回调。"""
    from dash import dcc

    return dcc.Tab(
        label='📖 使用说明',
        value='help',
        children=[
            html.Div([

                # ── 顶部提示横幅 ──────────────────────────────────────────────
                html.Div([
                    html.Span('💡', style={'fontSize': '20px', 'marginRight': '10px'}),
                    html.Span(
                        '本系统使用深度学习特征（DINOv3）对陶片图像进行自动聚类，'
                        '再通过多种可视化工具辅助考古学家开展器物归类与地层分析。'
                        '建议首次使用时按照下方步骤顺序浏览各功能模块。',
                        style={**_P, 'marginBottom': '0', 'color': '#1a3a6b'}
                    ),
                ], style={
                    'display': 'flex', 'alignItems': 'flex-start',
                    'padding': '14px 18px',
                    'backgroundColor': '#e8f0fb',
                    'borderRadius': '8px',
                    'border': '1px solid #c0d4f0',
                    'marginBottom': '16px',
                }),

                # ── 快速上手流程 ──────────────────────────────────────────────
                _section('快速上手流程', [
                    _step(1, '选择聚类参数',
                          '在页面顶部设置聚类数量（K）、算法和模式，点击【重新聚类】等待结果更新。'
                          '推荐从 K=15～25 开始尝试，K-Means 速度最快适合初步探索。'),
                    _step(2, '查看代表样本',
                          '打开【代表样本】标签，每个簇展示 5 张最典型图片。'
                          '确认各簇的器物形态是否具有考古意义。'),
                    _step(3, '散点图定位',
                          '在【散点图】中可按地层、部位、类型筛选，点击图中数据点可在右侧查看陶片图像。'),
                    _step(4, '地层流动分析',
                          '打开【地层流动】，查看各地层中的簇分布。'
                          'Sankey 图揭示哪些器物群贯穿多个地层（长期器型），哪些集中于单层（单次堆积）。'),
                    _step(5, '共现分析',
                          '打开【共现分析】，了解哪些簇倾向于同时出现在同一地层。'
                          '高共现的簇组合可能代表同一功能的器物组合，或同时代的器物群。'),
                ], style=_CARD_ACCENT),

                # ── 各标签页说明表格 ──────────────────────────────────────────
                html.Div([
                    html.H2('各功能模块说明', style=_H2),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th('标签页', style=_TH),
                            html.Th('主要用途', style=_TH),
                            html.Th('核心操作', style=_TH),
                            html.Th('解读要点', style=_TH),
                        ])),
                        html.Tbody([
                            _tab_row('代表样本',
                                     '直观检查每个簇的典型器物',
                                     '点击图片可加入比较面板；调整每页显示数量',
                                     '若某簇的 5 张图片形态差异很大，说明该簇内部混杂，可考虑增大 K 值'),
                            _tab_row('散点图',
                                     '在二维嵌入空间中定位所有陶片',
                                     '用筛选器按地层/部位/类型高亮；点击单个数据点查看图像',
                                     '相邻的点在特征空间中相近；跨簇紧密聚集的点可能是拼对候选'),
                            _tab_row('相似度',
                                     '查找与指定样本最相似的陶片',
                                     '输入样本 ID，设置返回数量',
                                     '结果跨簇出现时，说明该样本与相邻簇存在特征过渡'),
                            _tab_row('热力图',
                                     '查看簇间特征距离矩阵',
                                     '颜色深浅代表簇间平均距离',
                                     '颜色极浅的簇对说明两簇特征极为相近，可考虑合并'),
                            _tab_row('簇规模',
                                     '了解各簇的陶片数量分布',
                                     '查看直方图',
                                     '规模悬殊（个别簇过大或过小）时，说明 K 值可能需要调整'),
                            _tab_row('聚类质量',
                                     '评估当前聚类方案的好坏',
                                     '查看轮廓系数、Davies-Bouldin 等指标',
                                     '轮廓系数越高越好（最大 1）；Davis-Bouldin 越低越好'),
                            _tab_row('类别构成',
                                     '查看每个簇的器类/部位/地层构成比例',
                                     '切换横轴为"按簇"或"按地层"',
                                     '若某簇的器类构成与考古分类高度吻合，说明模型捕获了有意义的器物特征'),
                            _tab_row('簇分析',
                                     '查看簇质量详情和特征差异',
                                     '选择感兴趣的簇，调整 Top-K 特征数',
                                     '特征差异图中数值高的维度是该簇的"指纹特征"'),
                            _tab_row('地层流动',
                                     '分析器物群在地层序列中的分布规律',
                                     '调整最小连线阈值过滤低频关联；切换热力图归一化方式',
                                     '见下方【地层流动详解】'),
                            _tab_row('共现分析',
                                     '分析哪些器物群在同一地层中共同出现',
                                     '选择归一化方式；调整联接方法影响树状图分组',
                                     '见下方【共现分析详解】'),
                        ]),
                    ], style={'width': '100%', 'borderCollapse': 'collapse'}),
                ], style=_CARD),

                # ── 地层流动详解 ──────────────────────────────────────────────
                html.Div([
                    html.H2('地层流动详解', style=_H2),
                    html.Div([
                        html.Div([
                            html.H3('Sankey 图（流向图）', style=_H3),
                            html.P(
                                '左侧节点代表地层（⑭ 在顶部表示最新/最浅层，① 在底部表示最早/最深层）；'
                                '右侧节点代表簇。连线的粗细等于该层中属于该簇的陶片数量。',
                                style=_P),
                            html.P('解读方法：', style={**_P, 'fontWeight': '600', 'marginBottom': '2px'}),
                            html.Ul([
                                html.Li('某簇同时连向多个地层 → 该器型长期持续使用，时间跨度大', style=_P),
                                html.Li('某簇仅连向一两个地层 → 可能是某阶段特有器型或单次堆积事件', style=_P),
                                html.Li('多个簇的连线在某一层集中 → 该层出土器物种类丰富，可能是主要使用层', style=_P),
                            ], style={'paddingLeft': '18px', 'marginTop': '4px'}),
                        ], style={'flex': '1', 'minWidth': '260px', 'paddingRight': '16px'}),
                        html.Div([
                            html.H3('热力图归一化方式', style=_H3),
                            html.Table([
                                html.Thead(html.Tr([
                                    html.Th('模式', style=_TH),
                                    html.Th('含义', style=_TH),
                                    html.Th('适合分析', style=_TH),
                                ])),
                                html.Tbody([
                                    html.Tr([
                                        html.Td('绝对数', style=_TD),
                                        html.Td('该层中该簇的陶片数量', style=_TD),
                                        html.Td('了解各层出土规模', style=_TD),
                                    ]),
                                    html.Tr([
                                        html.Td('按层归一化', style=_TD),
                                        html.Td('该簇占该层总陶片的百分比', style=_TD),
                                        html.Td('比较各层的器物构成比例', style=_TD),
                                    ]),
                                    html.Tr([
                                        html.Td('按簇归一化', style=_TD),
                                        html.Td('该层占该簇总陶片的百分比', style=_TD),
                                        html.Td('了解某种器型主要集中在哪层', style=_TD),
                                    ]),
                                ]),
                            ], style={'width': '100%', 'borderCollapse': 'collapse'}),
                            html.H3('统计摘要说明', style=_H3),
                            html.Ul([
                                html.Li('跨层最广：出现地层数最多的簇，代表长期流行的器型', style=_P),
                                html.Li('最集中：Simpson 集中度最高，说明该器型几乎只出现在某一地层', style=_P),
                                html.Li('多样性（Shannon 熵）：0 表示该层只有一种簇，越接近 1 表示器物越多元', style=_P),
                            ], style={'paddingLeft': '18px', 'marginTop': '4px'}),
                        ], style={'flex': '1', 'minWidth': '260px'}),
                    ], style={'display': 'flex', 'gap': '0', 'flexWrap': 'wrap'}),
                ], style=_CARD_GREEN),

                # ── 共现分析详解 ──────────────────────────────────────────────
                html.Div([
                    html.H2('共现分析详解', style=_H2),
                    html.Div([
                        html.Div([
                            html.H3('什么是共现？', style=_H3),
                            html.P(
                                '如果簇 A 和簇 B 经常同时出现在同一个地层单位中，则两者具有高共现性。'
                                '高共现可能意味着：',
                                style=_P),
                            html.Ul([
                                html.Li('两种器型属于同一功能组合（如饮食器具 + 储藏器具同时使用）', style=_P),
                                html.Li('两种器型属于同一时代，随地层共同堆积', style=_P),
                                html.Li('来自同一制作传统或同一来源地', style=_P),
                            ], style={'paddingLeft': '18px', 'marginTop': '4px'}),
                            html.H3('归一化方式说明', style=_H3),
                            html.Table([
                                html.Thead(html.Tr([
                                    html.Th('方式', style=_TH),
                                    html.Th('公式', style=_TH),
                                    html.Th('特点', style=_TH),
                                ])),
                                html.Tbody([
                                    html.Tr([
                                        html.Td('原始计数', style=_TD),
                                        html.Td('A∩B 的层数', style=_TD),
                                        html.Td('反映绝对频率，规模大的簇偏高', style=_TD),
                                    ]),
                                    html.Tr([
                                        html.Td('Jaccard 相似度', style=_TD),
                                        html.Td('|A∩B| ÷ |A∪B|', style=_TD),
                                        html.Td('消除规模偏差，最推荐用于比较', style=_TD),
                                    ]),
                                    html.Tr([
                                        html.Td('条件概率 P(j|i)', style=_TD),
                                        html.Td('有 A 的层中 B 的出现率', style=_TD),
                                        html.Td('有方向性：A→B 与 B→A 不对称', style=_TD),
                                    ]),
                                ]),
                            ], style={'width': '100%', 'borderCollapse': 'collapse'}),
                        ], style={'flex': '1', 'minWidth': '260px', 'paddingRight': '16px'}),
                        html.Div([
                            html.H3('树状图解读', style=_H3),
                            html.P(
                                '树状图将共现行为相似的簇归为一组。树枝汇合点越低（距离越小），'
                                '说明这些簇共现关系越紧密。',
                                style=_P),
                            html.P('联接方法的选择：', style={**_P, 'fontWeight': '600', 'marginBottom': '2px'}),
                            html.Ul([
                                html.Li('Average（推荐）：取组间平均距离，结果稳健', style=_P),
                                html.Li('Complete：取组间最大距离，分组更紧凑', style=_P),
                                html.Li('Single：取组间最小距离，容易产生链式效应', style=_P),
                                html.Li('Ward：最小化组内方差，适合较规则的数据', style=_P),
                            ], style={'paddingLeft': '18px', 'marginTop': '4px'}),
                            html.H3('如何利用分析结果？', style=_H3),
                            html.P('建议工作流程：', style={**_P, 'fontWeight': '600', 'marginBottom': '2px'}),
                            _step('①', '用 Jaccard 模式找出强共现簇对（值 > 0.6）',
                                  ''),
                            _step('②', '在树状图中确认这些簇是否归为同一树枝',
                                  ''),
                            _step('③', '在散点图中查看这些簇的空间分布是否相邻',
                                  ''),
                            _step('④', '在代表样本中比较这些簇的器物图像，判断是否属于同一功能组',
                                  ''),
                        ], style={'flex': '1', 'minWidth': '260px'}),
                    ], style={'display': 'flex', 'gap': '0', 'flexWrap': 'wrap'}),
                ], style=_CARD_ACCENT),

                # ── 常见问题 ──────────────────────────────────────────────────
                html.Div([
                    html.H2('常见问题', style=_H2),
                    _faq('聚类结果中某些簇的图片看起来很混乱，怎么办？',
                         '通常有两种原因：① K 值设置过小，多种器型被强制归为一簇，可以尝试增大 K 值；'
                         '② 该簇中确实包含多种形态（如残片程度不一），可在簇分析标签页查看其轮廓系数，'
                         '低于 0 的簇内部分离度差。'),
                    _faq('地层流动的 Sankey 图显示"无满足阈值的连线"，如何处理？',
                         '将"Sankey 最小连线"滑块向左调小（如设为 1），或在层位筛选中选择陶片较多的地层。'
                         '该阈值的作用是过滤陶片数量过少的连线，避免图形过于复杂。'),
                    _faq('共现分析显示的"孤立簇"是什么？',
                         '在当前选定的层位范围内，该簇从未与其他簇同时出现在同一地层中。'
                         '可能原因：该器型仅出现于单一地层（时间极短）、或该器型在所选层位范围内样本极少。'
                         '尝试扩大层位筛选范围，或降低"最小共现层数"阈值。'),
                    _faq('如何判断当前 K 值是否合适？',
                         '打开【聚类质量】标签，观察轮廓系数（Silhouette Score）。'
                         '一般认为 > 0.3 为可接受，> 0.5 为较好。'
                         '同时参考【簇规模】标签，若出现规模极小（< 5 片）或极大（> 总量 30%）的簇，'
                         '说明 K 值需要调整。'),
                    _faq('散点图中的 UMAP 二维坐标代表什么？',
                         '每个点代表一张陶片图像，坐标由深度学习特征（128 维）降维而来，'
                         '空间距离近的点表示视觉特征相似。坐标本身无考古学单位意义，'
                         '仅用于相对位置的比较。'),
                ], style=_CARD_WARN),

            ], style={'padding': '16px', 'maxWidth': '1100px', 'margin': '0 auto'}),
        ],
    )


# ── 辅助构建函数 ──────────────────────────────────────────────────────────────

def _tab_row(tab, purpose, operation, tip):
    return html.Tr([
        html.Td(html.Span(tab, style={
            'fontWeight': '600', 'color': '#2c6fad', 'whiteSpace': 'nowrap',
        }), style=_TD),
        html.Td(purpose, style=_TD),
        html.Td(operation, style=_TD),
        html.Td(tip, style={**_TD, 'color': '#555'}),
    ])


def _faq(question, answer):
    return html.Div([
        html.Div([
            html.Span('Q', style={
                'display': 'inline-block', 'width': '20px', 'height': '20px',
                'borderRadius': '50%', 'backgroundColor': '#e8a838', 'color': '#fff',
                'fontSize': '12px', 'fontWeight': '700', 'textAlign': 'center',
                'lineHeight': '20px', 'marginRight': '8px', 'flexShrink': '0',
            }),
            html.Span(question, style={'fontWeight': '600', 'fontSize': '13px', 'color': '#222'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '4px'}),
        html.Div(answer, style={**_P, 'paddingLeft': '28px', 'color': '#444'}),
    ], style={'marginBottom': '14px'})
