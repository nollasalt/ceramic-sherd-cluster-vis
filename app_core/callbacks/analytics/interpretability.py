"""聚类可解释性分析回调：特征重要性、判别特征、簇轮廓。"""

import dash
from dash import Input, Output, State, html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from app_core.data_cache import get_data_cache
from app_core.utils import extract_cluster_visual_profile, analyze_cluster_feature_distribution
from performance_utils import cache_plot_result


def generate_cluster_explanation(visual_profile, cluster_df, cluster_col,
                                distribution_analysis=None, part_distribution=None):
    """生成聚类依据的解释性文字。"""
    parts = []

    # 颜色描述
    v = visual_profile['mean_hsv'][2]
    if v < 30:
        color_desc = "深色"
    elif v < 60:
        color_desc = "中等色调"
    else:
        color_desc = "浅色"

    # 装饰技法
    decoration = visual_profile.get('decoration_type', '未知')

    # 部位分布
    part_desc = ""
    if part_distribution and len(part_distribution) > 0:
        total = sum(part_distribution.values())
        dominant_part = max(part_distribution.items(), key=lambda x: x[1])
        part_ratio = dominant_part[1] / total
        if part_ratio > 0.6:
            part_desc = f"，主要为{dominant_part[0]}（{part_ratio*100:.0f}%）"
        elif len(part_distribution) > 1:
            top2 = sorted(part_distribution.items(), key=lambda x: x[1], reverse=True)[:2]
            part_desc = f"，包含{top2[0][0]}（{top2[0][1]/total*100:.0f}%）、{top2[1][0]}（{top2[1][1]/total*100:.0f}%）等"

    # 生成基础描述
    explanation = f"该簇陶片呈现{color_desc}特征，装饰技法为{decoration}{part_desc}。\n\n"

    # 分析主要决定因素
    if distribution_analysis:
        color_dist = distribution_analysis.get('color_distribution', {})
        decoration_dist = distribution_analysis.get('decoration_distribution', {})
        n = distribution_analysis.get('n_samples', 0)

        # 计算一致性（最大类别占比）
        color_consistency = max(color_dist.values()) / n if color_dist and n > 0 else 0
        decoration_consistency = max(decoration_dist.values()) / n if decoration_dist and n > 0 else 0
        part_consistency = dominant_part[1] / total if part_distribution and total > 0 else 0

        # 确定主要决定因素
        factors = []
        if decoration_consistency > 0.7:
            factors.append(f"装饰技法（{decoration_consistency*100:.0f}%一致）")
        if color_consistency > 0.7:
            factors.append(f"颜色特征（{color_consistency*100:.0f}%一致）")
        if part_consistency > 0.7:
            factors.append(f"部位类型（{part_consistency*100:.0f}%一致）")

        if factors:
            explanation += f"主要决定因素：{' + '.join(factors)}。"
        else:
            explanation += "主要决定因素：综合视觉特征的相似性（无单一主导因素）。"
    else:
        # 简化版解释
        if "素面" in decoration:
            explanation += "聚类依据可能是相似的表面处理方式和制作工艺。"
        elif "绳纹" in decoration or "篮纹" in decoration:
            explanation += "聚类依据可能是相似的装饰技法和纹饰风格。"
        elif "刻划" in decoration:
            explanation += "聚类依据可能是相似的刻划工艺和纹饰复杂度。"
        else:
            explanation += "聚类依据可能是综合的视觉特征相似性。"

    return explanation


def register_interpretability_callbacks(app, image_root=None):

    # 初始化簇选择下拉框
    @app.callback(
        Output('interp-cluster-select', 'options'),
        Output('interp-cluster-select', 'value'),
        Input('visualization-tabs', 'value'),
        State('data-store', 'data'),
    )
    def init_cluster_select(tab_value, _data_store):
        """当切换到可解释性标签页时填充簇选项。"""
        if tab_value != 'interpretability':
            return dash.no_update, dash.no_update

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']

        clusters = sorted(df[cluster_col].dropna().unique())
        opts = [{'label': f'簇 {c}', 'value': c} for c in clusters]
        default = clusters[0] if clusters else None

        return opts, default

    # 主渲染回调
    @app.callback(
        Output('interp-distribution-charts', 'figure'),
        Output('interp-profile', 'children'),
        Input('visualization-tabs', 'value'),
        Input('interp-cluster-select', 'value'),
        State('data-store', 'data'),
    )
    @cache_plot_result
    def render_interpretability(tab_value, selected_cluster, _data_store):
        """渲染可解释性分析图表。"""
        if tab_value != 'interpretability' or selected_cluster is None:
            return dash.no_update, dash.no_update

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']
        feature_cols = data_cache['feature_cols']

        if len(feature_cols) == 0:
            empty = px.scatter(title='无特征数据')
            return empty, html.Div('无特征数据', style={'color': '#666'})

        # 创建工作DataFrame，包含必要的列
        cols_to_include = ['sample_id', cluster_col] + feature_cols
        if 'part_C' in df.columns:
            cols_to_include.append('part_C')
        if 'unit_C' in df.columns:
            cols_to_include.append('unit_C')
        work = df[cols_to_include].dropna(subset=[cluster_col]).copy()

        cluster_mask = work[cluster_col] == selected_cluster
        cluster_data = work.loc[cluster_mask, feature_cols]

        if len(cluster_data) == 0:
            empty = px.scatter(title='该簇无数据')
            return empty, html.Div('该簇无数据', style={'color': '#666'})

        # ── 簇轮廓摘要 ────────────────────────────────────────────────────
        profile_items = [
            html.Div([
                html.Span('样本数', style={'color': '#666', 'fontSize': '12px'}),
                html.Div(f'{len(cluster_data)} 片', style={'fontWeight': '600', 'fontSize': '15px', 'color': '#1f77b4'}),
            ], style={'marginBottom': '10px'}),
        ]

        # 初始化变量
        distribution_analysis = None
        part_distribution = {}
        unit_distribution = {}

        # 分析部位分布（不依赖图像）
        cluster_df = work[work[cluster_col] == selected_cluster]
        if 'part_C' in cluster_df.columns:
            part_counts = cluster_df['part_C'].value_counts()
            part_distribution = {k: int(v) for k, v in part_counts.items()}
            print(f"[DEBUG] 部位分布: {part_distribution}")
        else:
            print(f"[DEBUG] 数据中没有 part_C 列，可用列: {cluster_df.columns.tolist()}")

        # 分析地层分布
        if 'unit_C' in cluster_df.columns:
            unit_counts = cluster_df['unit_C'].value_counts()
            unit_distribution = {k: int(v) for k, v in unit_counts.items()}

        # 提取视觉特征和分布分析
        if image_root:
            cluster_sample_ids = work.loc[cluster_mask, 'sample_id'].tolist()
            image_col_dict = df.set_index('sample_id')[data_cache['image_col']].to_dict()
            search_dirs = [Path(image_root)]

            visual_profile = extract_cluster_visual_profile(
                cluster_sample_ids, image_col_dict, search_dirs, max_samples=10
            )

            # 分析特征分布
            distribution_analysis = analyze_cluster_feature_distribution(
                cluster_sample_ids, image_col_dict, search_dirs, max_samples=50
            )

            if visual_profile:
                r, g, b = visual_profile['mean_rgb']
                h, s, v = visual_profile['mean_hsv']

                # 生成聚类依据解释（包含分布数据）
                explanation = generate_cluster_explanation(
                    visual_profile, cluster_df, cluster_col,
                    distribution_analysis, part_distribution
                )

                profile_items.extend([
                    html.Div([
                        html.Span('推断装饰技法', style={'color': '#666', 'fontSize': '12px'}),
                        html.Div(visual_profile.get('decoration_type', '未知'),
                                style={'fontWeight': '700', 'fontSize': '16px', 'color': '#e74c3c',
                                       'padding': '6px 10px', 'backgroundColor': '#fff5f5',
                                       'borderRadius': '4px', 'border': '1px solid #fadbd8'}),
                    ], style={'marginBottom': '12px'}),
                    html.Div([
                        html.Span('聚类依据解释', style={'color': '#666', 'fontSize': '12px'}),
                        html.Div(explanation,
                                style={'fontSize': '13px', 'color': '#555', 'lineHeight': '1.6',
                                       'padding': '8px 10px', 'backgroundColor': '#f8f9fa',
                                       'borderRadius': '4px', 'border': '1px solid #e9ecef'}),
                    ], style={'marginBottom': '12px'}),
                ])

                # 添加特征分布统计
                if distribution_analysis:
                    color_dist = distribution_analysis.get('color_distribution', {})
                    decoration_dist = distribution_analysis.get('decoration_distribution', {})
                    n = distribution_analysis.get('n_samples', 0)

                    dist_items = [html.Div('特征分布统计', style={'fontWeight': '600', 'fontSize': '14px', 'marginBottom': '8px'})]

                    # 颜色分布
                    if color_dist:
                        color_items = [html.Span(f"{k}: {v}片 ({v/n*100:.0f}%)", style={'marginRight': '12px'})
                                      for k, v in sorted(color_dist.items(), key=lambda x: x[1], reverse=True)]
                        dist_items.append(html.Div([
                            html.Span('颜色分布：', style={'color': '#666', 'fontSize': '12px', 'marginRight': '8px'}),
                            *color_items
                        ], style={'marginBottom': '6px', 'fontSize': '12px'}))

                    # 装饰技法分布
                    if decoration_dist:
                        deco_items = [html.Span(f"{k}: {v}片 ({v/n*100:.0f}%)", style={'marginRight': '12px'})
                                     for k, v in sorted(decoration_dist.items(), key=lambda x: x[1], reverse=True)]
                        dist_items.append(html.Div([
                            html.Span('装饰技法：', style={'color': '#666', 'fontSize': '12px', 'marginRight': '8px'}),
                            *deco_items
                        ], style={'marginBottom': '6px', 'fontSize': '12px'}))

                    # 部位分布
                    if part_distribution:
                        total = sum(part_distribution.values())
                        part_items = [html.Span(f"{k}: {v}片 ({v/total*100:.0f}%)", style={'marginRight': '12px'})
                                     for k, v in sorted(part_distribution.items(), key=lambda x: x[1], reverse=True)]
                        dist_items.append(html.Div([
                            html.Span('部位分布：', style={'color': '#666', 'fontSize': '12px', 'marginRight': '8px'}),
                            *part_items
                        ], style={'marginBottom': '6px', 'fontSize': '12px'}))

                    profile_items.append(html.Div(dist_items,
                                                  style={'padding': '8px 10px', 'backgroundColor': '#f0f8ff',
                                                         'borderRadius': '4px', 'border': '1px solid #d0e8ff',
                                                         'marginBottom': '12px'}))

                profile_items.extend([
                    html.Div([
                        html.Span('平均颜色', style={'color': '#666', 'fontSize': '12px'}),
                        html.Div([
                            html.Div(style={
                                'width': '40px', 'height': '20px', 'display': 'inline-block',
                                'backgroundColor': f'rgb({int(r)},{int(g)},{int(b)})',
                                'border': '1px solid #ccc', 'marginRight': '8px', 'verticalAlign': 'middle'
                            }),
                            html.Span(f'RGB({int(r)}, {int(g)}, {int(b)})',
                                     style={'fontSize': '13px', 'color': '#333', 'verticalAlign': 'middle'}),
                        ]),
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Span('色调/饱和度/明度', style={'color': '#666', 'fontSize': '12px'}),
                        html.Div(f'H:{h:.0f}° S:{s:.0f}% V:{v:.0f}%',
                                style={'fontWeight': '600', 'fontSize': '14px', 'color': '#333'}),
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Span('亮度', style={'color': '#666', 'fontSize': '12px'}),
                        html.Div(f'{visual_profile["mean_brightness"]:.1f} ± {visual_profile["std_brightness"]:.1f}',
                                style={'fontWeight': '600', 'fontSize': '14px', 'color': '#333'}),
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Span('对比度', style={'color': '#666', 'fontSize': '12px'}),
                        html.Div(f'{visual_profile["mean_contrast"]:.1f} ± {visual_profile["std_contrast"]:.1f}',
                                style={'fontWeight': '600', 'fontSize': '14px', 'color': '#333'}),
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Span('纹理复杂度', style={'color': '#666', 'fontSize': '12px'}),
                        html.Div(f'{visual_profile["mean_texture"]:.1f} ± {visual_profile["std_texture"]:.1f}',
                                style={'fontWeight': '600', 'fontSize': '14px', 'color': '#333'}),
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Span('GLCM对比度（纹理变化强度）', style={'color': '#666', 'fontSize': '12px'}),
                        html.Div(f'{visual_profile["mean_glcm_contrast"]:.2f} ± {visual_profile["std_glcm_contrast"]:.2f}',
                                style={'fontWeight': '600', 'fontSize': '14px', 'color': '#333'}),
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Span('GLCM同质性（纹理均匀度）', style={'color': '#666', 'fontSize': '12px'}),
                        html.Div(f'{visual_profile["mean_glcm_homogeneity"]:.3f} ± {visual_profile["std_glcm_homogeneity"]:.3f}',
                                style={'fontWeight': '600', 'fontSize': '14px', 'color': '#333'}),
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Span('GLCM能量（纹理有序性）', style={'color': '#666', 'fontSize': '12px'}),
                        html.Div(f'{visual_profile["mean_glcm_energy"]:.3f} ± {visual_profile["std_glcm_energy"]:.3f}',
                                style={'fontWeight': '600', 'fontSize': '14px', 'color': '#333'}),
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Span('GLCM相关性（纹理方向性）', style={'color': '#666', 'fontSize': '12px'}),
                        html.Div(f'{visual_profile["mean_glcm_correlation"]:.3f} ± {visual_profile["std_glcm_correlation"]:.3f}',
                                style={'fontWeight': '600', 'fontSize': '14px', 'color': '#333'}),
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Span('GLCM熵（纹理随机性）', style={'color': '#666', 'fontSize': '12px'}),
                        html.Div(f'{visual_profile["mean_glcm_entropy"]:.2f} ± {visual_profile["std_glcm_entropy"]:.2f}',
                                style={'fontWeight': '600', 'fontSize': '14px', 'color': '#333'}),
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Span(f'基于 {visual_profile["n_samples"]} 张图像',
                                 style={'fontSize': '11px', 'color': '#999', 'fontStyle': 'italic'}),
                    ]),
                ])

        # 创建特征分布图表
        from plotly.subplots import make_subplots

        dist_fig = make_subplots(
            rows=1, cols=4,
            subplot_titles=('装饰技法分布', '颜色分布', '部位分布', '地层分布'),
            specs=[[{'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}]]
        )

        if distribution_analysis:
            # 装饰技法分布
            decoration_dist = distribution_analysis.get('decoration_distribution', {})
            if decoration_dist:
                dist_fig.add_trace(
                    go.Pie(labels=list(decoration_dist.keys()), values=list(decoration_dist.values()),
                           name='装饰技法', hole=0.3),
                    row=1, col=1
                )

            # 颜色分布
            color_dist = distribution_analysis.get('color_distribution', {})
            if color_dist:
                dist_fig.add_trace(
                    go.Pie(labels=list(color_dist.keys()), values=list(color_dist.values()),
                           name='颜色', hole=0.3),
                    row=1, col=2
                )

        # 部位分布（只显示Top-5，其他合并）
        if part_distribution:
            sorted_parts = sorted(part_distribution.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_parts) > 5:
                top5 = dict(sorted_parts[:5])
                other_count = sum(v for k, v in sorted_parts[5:])
                if other_count > 0:
                    top5['其他'] = other_count
                part_dist_display = top5
            else:
                part_dist_display = part_distribution

            dist_fig.add_trace(
                go.Pie(labels=list(part_dist_display.keys()), values=list(part_dist_display.values()),
                       name='部位', hole=0.3),
                row=1, col=3
            )

        # 地层分布（只显示Top-5，其他合并）
        if unit_distribution:
            sorted_units = sorted(unit_distribution.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_units) > 5:
                top5 = dict(sorted_units[:5])
                other_count = sum(v for k, v in sorted_units[5:])
                if other_count > 0:
                    top5['其他'] = other_count
                unit_dist_display = top5
            else:
                unit_dist_display = unit_distribution

            dist_fig.add_trace(
                go.Pie(labels=list(unit_dist_display.keys()), values=list(unit_dist_display.values()),
                       name='地层', hole=0.3),
                row=1, col=4
            )

        dist_fig.update_layout(
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=20),
            height=350
        )

        return dist_fig, html.Div(profile_items)
