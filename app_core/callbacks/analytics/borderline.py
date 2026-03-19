"""边界样本识别回调。

边界度 = 样本到本簇中心的距离 / 样本到最近其他簇中心的距离。
分数 → 1 或 > 1 时，样本处于两簇交界区，是拼对候选、分类错误或独特器物的首选排查对象。
"""
from pathlib import Path

import dash
from dash import Input, Output, State, html
import numpy as np
import pandas as pd

from app_core.data_cache import get_data_cache


# ── 辅助函数 ───────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    """根据边界度分数返回颜色（红/橙/黄绿）。"""
    if score >= 0.90:
        return '#e74c3c'
    if score >= 0.70:
        return '#e67e22'
    if score >= 0.50:
        return '#f0b429'
    return '#27ae60'


def _score_bar(score: float, color: str) -> html.Div:
    """渲染细长进度条代表边界度，宽度上限 100%（score 可能 > 1）。"""
    pct = min(int(score * 100), 100)
    return html.Div(
        html.Div(style={
            'width': f'{pct}%', 'height': '4px',
            'backgroundColor': color, 'borderRadius': '2px',
        }),
        style={
            'width': '100%', 'height': '4px',
            'backgroundColor': '#eee', 'borderRadius': '2px',
            'marginTop': '4px',
        },
    )


def _compute_borderline_scores(work: pd.DataFrame, cluster_col: str, feature_cols: list):
    """
    返回 work 的副本，新增列：
      _d_self          - 到本簇中心的 L2 距离
      _d_nearest_other - 到最近其他簇中心的 L2 距离
      _nearest_other   - 最近其他簇的 cluster_col 值
      _bscore          - 边界度分数 = _d_self / _d_nearest_other
    """
    clusters = sorted(work[cluster_col].dropna().unique())
    if len(clusters) < 2 or not feature_cols:
        work = work.copy()
        work['_d_self'] = np.nan
        work['_d_nearest_other'] = np.nan
        work['_nearest_other'] = None
        work['_bscore'] = np.nan
        return work

    # 计算所有簇中心
    centers = {}
    for c in clusters:
        mask = work[cluster_col] == c
        if mask.sum() > 0:
            centers[c] = work.loc[mask, feature_cols].mean().values

    if len(centers) < 2:
        work = work.copy()
        for col in ('_d_self', '_d_nearest_other', '_bscore'):
            work[col] = np.nan
        work['_nearest_other'] = None
        return work

    center_ids = list(centers.keys())
    C = np.array([centers[c] for c in center_ids], dtype=np.float32)  # (K, F)
    X = work[feature_cols].values.astype(np.float32)  # (N, F)

    # 高维时先 PCA 降维，减少矩阵乘法开销（128→50 维约快 60%）
    if X.shape[1] > 50:
        from sklearn.decomposition import PCA as _PCA
        n_comp = min(50, X.shape[0] - 1, X.shape[1])
        _pca = _PCA(n_components=n_comp, random_state=42)
        X = _pca.fit_transform(X).astype(np.float32)
        C = _pca.transform(C).astype(np.float32)

    # 高效计算 (N, K) 距离矩阵：||x - c||^2 = ||x||^2 - 2 x·cᵀ + ||c||^2
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)   # (N, 1)
    C_sq = np.sum(C ** 2, axis=1, keepdims=True).T  # (1, K)
    dists_sq = np.maximum(X_sq - 2.0 * (X @ C.T) + C_sq, 0.0)  # (N, K)
    dists = np.sqrt(dists_sq)  # (N, K)

    c_to_idx = {c: i for i, c in enumerate(center_ids)}
    own_labels = work[cluster_col].values

    # 向量化：将每行的"自簇"列设为 inf，再 argmin
    own_idx_arr = np.array([c_to_idx.get(lbl, -1) for lbl in own_labels])  # (N,)
    valid_mask = own_idx_arr >= 0

    # d_self（向量化取对角）
    safe_own = np.where(own_idx_arr >= 0, own_idx_arr, 0)
    d_self = dists[np.arange(len(work)), safe_own].astype(float)
    d_self[~valid_mask] = np.nan

    # 屏蔽自簇列后 argmin
    dists_other = dists.copy()
    dists_other[np.arange(len(work))[valid_mask], own_idx_arr[valid_mask]] = np.inf
    best_idx = np.argmin(dists_other, axis=1)  # (N,)

    d_nearest_other = dists_other[np.arange(len(work)), best_idx].astype(float)
    d_nearest_other[~valid_mask] = np.nan
    d_nearest_other[d_nearest_other == np.inf] = np.nan

    nearest_other = np.array([center_ids[i] for i in best_idx], dtype=object)
    nearest_other[~valid_mask] = None

    bscore = d_self / (d_nearest_other + 1e-8)

    result = work.copy()
    result['_d_self'] = d_self
    result['_d_nearest_other'] = d_nearest_other
    result['_nearest_other'] = nearest_other
    result['_bscore'] = bscore
    return result


# ── 主回调 ────────────────────────────────────────────────────────────────────

def register_borderline_callbacks(app, *, image_root):
    base_root = Path(__file__).parent.parent.parent.parent

    image_root_abs = Path(image_root)
    if not image_root_abs.is_absolute():
        image_root_abs = base_root / image_root_abs

    def resolve_path(val: str):
        """解析图像路径，与 representatives 保持一致的查找顺序。"""
        p = Path(str(val))
        if not p.is_absolute():
            p = image_root_abs / p
        if p.exists():
            return p
        for subdir in ('all_cutouts', 'all_kmeans_new'):
            alt = base_root / subdir / p.name
            if alt.exists():
                return alt
        return p

    @app.callback(
        [Output('borderline-grid', 'children'),
         Output('borderline-pair-stats', 'children'),
         Output('borderline-page-index', 'data'),
         Output('borderline-page-status', 'children'),
         Output('borderline-page-prev', 'disabled'),
         Output('borderline-page-next', 'disabled')],
        [Input('visualization-tabs', 'value'),
         Input('borderline-per-cluster', 'value'),
         Input('borderline-threshold', 'value'),
         Input('borderline-page-prev', 'n_clicks'),
         Input('borderline-page-next', 'n_clicks'),
         Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value')],
        State('borderline-page-index', 'data'),
        State('data-store', 'data'),
    )
    def render_borderline(
        tab_value, per_cluster, threshold,
        prev_clicks, next_clicks,
        selected_clusters, selected_units, selected_parts, selected_types,
        page_index, data_store,
    ):
        """渲染边界样本图格与模糊边界对统计。"""
        if tab_value != 'borderline':
            return (dash.no_update,) * 6

        page_size = 8
        page_index = max(1, int(page_index or 1))

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']
        feature_cols = data_cache.get('feature_cols', [])
        image_col = data_cache['image_col']

        dff = df.copy()
        if selected_clusters:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        if selected_units and 'unit_C' in dff.columns:
            dff = dff[dff['unit_C'].isin(selected_units)]
        if selected_parts and 'part_C' in dff.columns:
            dff = dff[dff['part_C'].isin(selected_parts)]
        if selected_types and 'type_C' in dff.columns:
            dff = dff[dff['type_C'].isin(selected_types)]

        _empty = html.Div('暂无数据', style={'color': '#888', 'padding': '8px'})

        if not feature_cols or cluster_col not in dff.columns or len(dff) == 0:
            return _empty, _empty, 1, '第 0/0 页（0 个簇）', True, True

        # ── 计算边界度分数 ────────────────────────────────────────────────
        work = dff.dropna(subset=feature_cols + [cluster_col])
        if len(work) < 4 or len(work[cluster_col].unique()) < 2:
            return _empty, _empty, 1, '第 0/0 页（簇数不足）', True, True

        scored = _compute_borderline_scores(work, cluster_col, feature_cols)
        threshold = float(threshold or 0.5)

        # ── 分页 ─────────────────────────────────────────────────────────
        clusters = sorted(scored[cluster_col].dropna().unique())
        total_pages = max(1, (len(clusters) + page_size - 1) // page_size)

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        if trigger_id == 'borderline-page-prev':
            page_index = max(1, page_index - 1)
        elif trigger_id == 'borderline-page-next':
            page_index = min(total_pages, page_index + 1)
        else:
            page_index = 1

        page_index = max(1, min(page_index, total_pages))
        start_idx = (page_index - 1) * page_size
        active_clusters = clusters[start_idx:start_idx + page_size]

        per_cluster = max(1, min(8, int(per_cluster or 3)))
        thumb_size = 100

        # ── 渲染每个簇的边界样本卡片 ──────────────────────────────────────
        cluster_blocks = []
        for c in active_clusters:
            cdf = scored[scored[cluster_col] == c].copy()
            cdf = cdf[cdf['_bscore'] >= threshold].sort_values('_bscore', ascending=False)
            top_samples = cdf.head(per_cluster)

            if len(top_samples) == 0:
                cluster_blocks.append(html.Div([
                    html.Div(f'簇 {c}', style={
                        'fontSize': '13px', 'fontWeight': '700', 'marginBottom': '8px',
                    }),
                    html.Div('当前阈值下无边界样本', style={
                        'fontSize': '12px', 'color': '#999', 'padding': '8px 0',
                    }),
                ], style={
                    'padding': '10px', 'border': '1px solid #e4e8ef',
                    'borderRadius': '8px', 'backgroundColor': '#fafbfc',
                    'minWidth': '160px',
                }))
                continue

            cards = []
            for _, row in top_samples.iterrows():
                score = float(row['_bscore'])
                nearest = row.get('_nearest_other')
                d_self = row.get('_d_self', float('nan'))
                color = _score_color(score)

                img_val = row.get('image_name') if 'image_name' in row else row.get(image_col)
                fname = Path(resolve_path(img_val)).name

                score_label = html.Span(
                    f'{score:.2f}',
                    style={
                        'fontSize': '11px', 'fontWeight': '700', 'color': color,
                        'backgroundColor': color + '18',
                        'padding': '1px 5px', 'borderRadius': '4px',
                        'border': f'1px solid {color}44',
                    },
                )
                nearest_badge = html.Span(
                    f'→ 簇{nearest}',
                    style={
                        'fontSize': '10px', 'color': '#2c6fad',
                        'backgroundColor': '#e8f0fa',
                        'padding': '1px 5px', 'borderRadius': '4px',
                        'border': '1px solid #c0d4ed',
                        'marginLeft': '4px',
                    },
                ) if nearest is not None else html.Span('')

                cards.append(html.Div([
                    html.Img(
                        src=f'/img/{fname}',
                        style={
                            'width': '100%', 'height': f'{thumb_size}px',
                            'objectFit': 'cover', 'borderRadius': '6px 6px 0 0',
                            'border': '1px solid #e0e0e0',
                        },
                        title=str(img_val),
                        **{'data-image-path': fname},
                    ),
                    html.Div([
                        html.Div(
                            f"#{row.get('sample_id', '?')}",
                            style={
                                'fontSize': '10px', 'color': '#555', 'fontWeight': '600',
                                'overflow': 'hidden', 'textOverflow': 'ellipsis',
                                'whiteSpace': 'nowrap', 'marginBottom': '3px',
                            },
                        ),
                        html.Div([score_label, nearest_badge],
                                 style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '2px'}),
                        _score_bar(score, color),
                    ], style={'padding': '5px 6px'}),
                ], style={
                    'width': f'{thumb_size + 10}px',
                    'border': f'1px solid {color}44',
                    'borderRadius': '8px',
                    'backgroundColor': '#fff',
                    'boxShadow': f'0 1px 4px {color}22',
                }))

            cluster_blocks.append(html.Div([
                html.Div([
                    html.Span(f'簇 {c}', style={'fontSize': '13px', 'fontWeight': '700'}),
                    html.Span(
                        f'{len(cdf)} 个边界样本',
                        style={
                            'fontSize': '11px', 'color': '#888',
                            'marginLeft': '8px',
                        },
                    ),
                ], style={'marginBottom': '8px'}),
                html.Div(cards, style={'display': 'flex', 'gap': '6px', 'flexWrap': 'wrap'}),
            ], style={
                'padding': '10px 12px',
                'border': '1px solid #e0e0e0',
                'borderRadius': '8px',
                'backgroundColor': '#fff',
                'minWidth': '200px',
            }))

        page_status = (
            f'第 {page_index}/{total_pages} 页｜'
            f'簇 {start_idx + 1}–{min(start_idx + page_size, len(clusters))} / {len(clusters)}'
        )
        prev_disabled = page_index <= 1
        next_disabled = page_index >= total_pages

        # ── 模糊边界对统计 ────────────────────────────────────────────────
        valid = scored.dropna(subset=['_bscore', '_nearest_other'])
        valid = valid[valid['_bscore'] >= threshold]

        _a = valid[cluster_col].values
        _b = valid['_nearest_other'].values
        _not_same = _a != _b
        _a, _b = _a[_not_same], _b[_not_same]
        if len(_a) == 0:
            pair_counts = {}
        else:
            _pair_df = pd.DataFrame({'ca': np.minimum(_a, _b), 'cb': np.maximum(_a, _b)})
            pair_counts = dict(_pair_df.groupby(['ca', 'cb']).size())

        if not pair_counts:
            pair_stats_div = html.Div(
                '当前阈值下无显著模糊边界对。',
                style={'color': '#888', 'fontSize': '12px'},
            )
        else:
            top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:10]
            max_count = top_pairs[0][1]
            pair_rows = []
            for (ca, cb), cnt in top_pairs:
                bar_pct = int(cnt / max_count * 100)
                pair_rows.append(html.Div([
                    html.Div(
                        f'簇 {ca} ↔ 簇 {cb}',
                        style={'fontSize': '12px', 'fontWeight': '600', 'color': '#333',
                               'minWidth': '90px', 'flexShrink': '0'},
                    ),
                    html.Div(
                        html.Div(style={
                            'width': f'{bar_pct}%', 'height': '100%',
                            'backgroundColor': '#2c6fad', 'borderRadius': '3px',
                        }),
                        style={
                            'flex': '1', 'height': '14px',
                            'backgroundColor': '#e8edf4', 'borderRadius': '3px',
                            'margin': '0 8px',
                        },
                    ),
                    html.Span(
                        f'{cnt} 片',
                        style={'fontSize': '11px', 'color': '#888', 'flexShrink': '0'},
                    ),
                ], style={
                    'display': 'flex', 'alignItems': 'center',
                    'marginBottom': '6px',
                }))

            pair_stats_div = html.Div([
                html.Div(pair_rows),
                html.Div(
                    '（数值越大，两簇越难区分，可优先合并比较或降低 K 值）',
                    style={'fontSize': '11px', 'color': '#aaa', 'marginTop': '6px'},
                ),
            ])

        return (
            cluster_blocks,
            pair_stats_div,
            page_index,
            page_status,
            prev_disabled,
            next_disabled,
        )
