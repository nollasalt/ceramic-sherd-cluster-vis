import plotly.express as px

# 离散形状序列，用于在散点图中区分陶片部位
PART_SYMBOL_SEQUENCE = [
    'circle', 'square', 'diamond', 'cross', 'x',
    'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right',
    'pentagon', 'hexagon', 'star', 'hexagram', 'star-square', 'star-diamond'
]

_color_cache = {}

def generate_distinct_colors(n_colors: int):
    """生成 n 个视觉上不同的颜色（带缓存）。"""
    if n_colors in _color_cache:
        return _color_cache[n_colors]

    base_colors = (
        px.colors.qualitative.Plotly +      # 10 colors
        px.colors.qualitative.D3 +          # 10 colors
        px.colors.qualitative.G10 +         # 10 colors
        px.colors.qualitative.T10 +         # 10 colors
        px.colors.qualitative.Alphabet      # 26 colors
    )

    colors = [base_colors[i % len(base_colors)] for i in range(n_colors)]
    _color_cache[n_colors] = colors
    return colors

# 预生成50种不同的颜色
CLUSTER_COLORS = generate_distinct_colors(50)


def get_part_symbol_settings(dataframe):
    """生成基于部位字段的形状映射。"""
    symbol_col = None
    if 'part_C' in dataframe.columns and dataframe['part_C'].notna().any():
        symbol_col = 'part_C'
    elif 'part' in dataframe.columns and dataframe['part'].notna().any():
        symbol_col = 'part'

    if symbol_col is None:
        return None, {}

    parts = [p for p in dataframe[symbol_col].dropna().unique()]
    parts = sorted(parts, key=lambda x: str(x))
    symbol_map = {p: PART_SYMBOL_SEQUENCE[i % len(PART_SYMBOL_SEQUENCE)] for i, p in enumerate(parts)}

    return symbol_col, symbol_map
