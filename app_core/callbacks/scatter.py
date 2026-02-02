from pathlib import Path
import json
import os

import dash
from dash import Input, Output, State, html
import pandas as pd
import numpy as np
import plotly.express as px

from app_core.data_cache import get_data_cache, set_data_cache
from app_core.utils import CLUSTER_COLORS, PART_SYMBOL_SEQUENCE, get_part_symbol_settings
from data_processing import (
    #detect_columns,
    ensure_dimensionality_reduction,
    #ensure_sample_ids,
    img_to_base64,
    img_to_base64_full,
    load_cluster_metadata,
)


def register_scatter_callbacks(app, *, csv_path, image_root, get_filter_options):
    """Register scatter/filters related callbacks."""

    @app.callback(
        [Output('unit-filter', 'options'),
         Output('part-filter', 'options'),
         Output('type-filter', 'options')],
        Input('cluster-filter', 'value')
    )
    def update_filter_options(selected_clusters):
        return get_filter_options(selected_clusters)

    @app.callback(
        Output('hover-state', 'data'),
        [Input('tsne-plot', 'hoverData')],
        [State('sample-cluster-mapping', 'data')]
    )
    def update_hover_state(hoverData, sample_cluster_mapping):
        if not hoverData or not sample_cluster_mapping:
            return {'hovered_cluster': None}
        try:
            hover_point = hoverData['points'][0]
            if 'customdata' in hover_point and hover_point['customdata']:
                sample_id = hover_point['customdata'][0]
                cluster_id = sample_cluster_mapping.get(sample_id)
                return {'hovered_cluster': cluster_id}
            return {'hovered_cluster': None}
        except Exception:
            return {'hovered_cluster': None}

    app.clientside_callback(
        """
        function(hover_state, figure) {
            if (!figure) return figure;

            const has3d = figure.data && figure.data.some(t => t.type === 'scatter3d');
            if (has3d) {
                return window.dash_clientside.no_update;
            }

            const hovered_cluster = hover_state ? hover_state.hovered_cluster : null;
            const new_figure = JSON.parse(JSON.stringify(figure));

            if (new_figure.data) {
                new_figure.data.forEach(trace => {
                    if (hovered_cluster !== null && hovered_cluster !== undefined) {
                        const name = trace.name || '';
                        const clusterMatch = name === String(hovered_cluster) || name.startsWith(String(hovered_cluster) + ',');
                        trace.opacity = clusterMatch ? 1.0 : 0.2;
                    } else {
                        trace.opacity = 0.85;
                    }
                });
            }

            return new_figure;
        }
        """,
        Output('tsne-plot', 'figure', allow_duplicate=True),
        [Input('hover-state', 'data')],
        [State('tsne-plot', 'figure')],
        prevent_initial_call=True
    )

    @app.callback(
        [Output('tsne-plot', 'figure'),
         Output('data-store', 'data'),
         Output('sample-cluster-mapping', 'data'),
         Output('cluster-filter', 'options')],
        [Input('cluster-filter', 'value'),
         Input('unit-filter', 'value'),
         Input('part-filter', 'value'),
         Input('type-filter', 'value'),
         Input('algorithm-selector', 'value'),
         Input('dimension-selector', 'value'),
         Input('z-axis-selector', 'value'),
         Input('reload-trigger', 'data')],
        State('data-store', 'data'),
        prevent_initial_call=True
    )
    def update_plot(selected_clusters, selected_units, selected_parts, selected_types,
                    selected_algorithm, selected_dimension, z_axis='dimension',
                    reload_trigger=0, data_store=None):
        data_cache = get_data_cache()
        current_feature_cols = data_cache['feature_cols']
        raw_feature_cols = data_cache.get('raw_feature_cols', current_feature_cols)
        old_cluster_mode = data_cache.get('cluster_mode', 'merged')

        # Handle dataset reload trigger to refresh server-side cache when reclustering completes
        ctx = dash.callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'reload-trigger.data' and reload_trigger > 0:
            new_metadata = load_cluster_metadata()
            new_cluster_mode = new_metadata.get('cluster_mode', 'merged') if new_metadata else 'merged'
            df_new = pd.read_csv(csv_path)
            #new_cluster_col, new_image_col = detect_columns(df_new)
            new_cluster_col = "cluster_id"
            new_image_col = "image_name"
            if new_cluster_col is None or new_image_col is None:
                raise RuntimeError('无法识别聚类列或图片列，请检查 CSV')
            #df_new = ensure_sample_ids(df_new, new_image_col)
            exclude = {new_cluster_col, new_image_col, 'image_name', 'sample_id', 'side', 'image_id', 'sherd_id',
                      'unit', 'part', 'type', 'image_side', 'image_id_original', 'unit_C', 'part_C', 'type_C', 'image_path'}
            new_raw_feature_cols = [c for c in df_new.columns if c not in exclude and np.issubdtype(df_new[c].dtype, np.number)]
            df_processed = df_new.copy()
            new_feature_cols = list(new_raw_feature_cols)
            data_cache = {
                'df': df_processed,
                'feature_cols': new_feature_cols,
                'raw_feature_cols': new_raw_feature_cols,
                'cluster_col': new_cluster_col,
                'image_col': new_image_col,
                'cluster_mode': new_cluster_mode,
                'version': reload_trigger
            }
            set_data_cache(data_cache)

        # Use the cached DataFrame directly instead of re-sending via the client store
        df = data_cache['df']
        feature_cols = data_cache['feature_cols']
        cluster_col = data_cache['cluster_col']
        image_col = data_cache['image_col']

        if selected_algorithm is None:
            selected_algorithm = 'umap'

        if selected_algorithm == 'tsne':
            df, reduction_key = ensure_dimensionality_reduction(
                df, feature_cols, algorithm=selected_algorithm, n_components=selected_dimension, perplexity=30
            )
        elif selected_algorithm == 'umap':
            df, reduction_key = ensure_dimensionality_reduction(
                df, feature_cols, algorithm=selected_algorithm, n_components=selected_dimension, n_neighbors=15, min_dist=0.1
            )
        else:
            df, reduction_key = ensure_dimensionality_reduction(
                df, feature_cols, algorithm=selected_algorithm, n_components=selected_dimension
            )

        part_symbol_col, part_symbol_map = get_part_symbol_settings(df)
        symbol_kwargs = {}
        if part_symbol_col:
            symbol_kwargs = {
                'symbol': part_symbol_col,
                'symbol_map': part_symbol_map,
                'symbol_sequence': PART_SYMBOL_SEQUENCE
            }

        symbol_kwargs_3d = {}
        dff = df.copy()

        if selected_clusters and len(selected_clusters) > 0:
            dff = dff[dff[cluster_col].isin(selected_clusters)]
        if selected_units and len(selected_units) > 0 and 'unit_C' in dff.columns:
            dff = dff[dff['unit_C'].isin(selected_units)]
        if selected_parts and len(selected_parts) > 0 and 'part_C' in dff.columns:
            dff = dff[dff['part_C'].isin(selected_parts)]
        if selected_types and len(selected_types) > 0 and 'type_C' in dff.columns:
            dff = dff[dff['type_C'].isin(selected_types)]

        hover_cols = ['sample_id', 'image_name']
        if 'sherd_id' in dff.columns:
            hover_cols.append('sherd_id')
        if 'unit_C' in dff.columns:
            hover_cols.append('unit_C')
        if 'part_C' in dff.columns:
            hover_cols.append('part_C')
        if 'type_C' in dff.columns:
            hover_cols.append('type_C')

        custom = ['sample_id', 'image_name']
        if 'paired_images' in dff.columns:
            custom.append('paired_images')
        if 'sherd_id' in dff.columns:
            custom.append('sherd_id')

        if selected_dimension == 2:
            fig = px.scatter(
                dff,
                x=f'{reduction_key}_0',
                y=f'{reduction_key}_1',
                color=dff[cluster_col].astype(str),
                hover_data=hover_cols,
                custom_data=custom,
                color_discrete_sequence=CLUSTER_COLORS,
                render_mode='webgl',
                **symbol_kwargs,
            )
        else:
            if z_axis == 'unit_C' and 'unit_C' in dff.columns:
                circle_to_num = {
                    '①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5,
                    '⑥': 6, '⑦': 7, '⑧': 8, '⑨': 9, '⑩': 10,
                    '⑪': 11, '⑫': 12, '⑬': 13, '⑭': 14, '⑮': 15,
                    '⑯': 16, '⑰': 17, '⑱': 18, '⑲': 19, '⑳': 20,
                    '㉑': 21, '㉒': 22, '㉓': 23, '㉔': 24, '㉕': 25,
                    '㉖': 26, '㉗': 27, '㉘': 28, '㉙': 29, '㉚': 30
                }

                dff = dff.copy()
                dff['h690_num'] = dff['unit_C'].apply(
                    lambda x: circle_to_num.get(x[4:], 0)
                    if isinstance(x, str) and x.startswith('H690') and len(x) > 4
                    else 0
                )

                fig = px.scatter_3d(
                    dff,
                    x=f'{reduction_key}_0',
                    y=f'{reduction_key}_1',
                    z='h690_num',
                    color=dff[cluster_col].astype(str),
                    hover_data=hover_cols + ['unit_C'],
                    custom_data=custom,
                    title=f'{selected_algorithm} + unit_C三维图',
                    color_discrete_sequence=CLUSTER_COLORS,
                    **symbol_kwargs_3d,
                )
                fig.update_layout(scene=dict(zaxis=dict(title='H690序号')))
                fig.update_traces(marker={'size': 1})
            else:
                df, reduction_key_3d = ensure_dimensionality_reduction(
                    df,
                    feature_cols,
                    algorithm=selected_algorithm,
                    n_components=3,
                    perplexity=30 if selected_algorithm == 'tsne' else None,
                    n_neighbors=15 if selected_algorithm == 'umap' else None,
                    min_dist=0.1 if selected_algorithm == 'umap' else None,
                )
                dff = df.copy()
                if selected_clusters and len(selected_clusters) > 0:
                    dff = dff[dff[cluster_col].isin(selected_clusters)]
                if selected_units and len(selected_units) > 0 and 'unit_C' in dff.columns:
                    dff = dff[dff['unit_C'].isin(selected_units)]
                if selected_parts and len(selected_parts) > 0 and 'part_C' in dff.columns:
                    dff = dff[dff['part_C'].isin(selected_parts)]
                if selected_types and len(selected_types) > 0 and 'type_C' in dff.columns:
                    dff = dff[dff['type_C'].isin(selected_types)]

                fig = px.scatter_3d(
                    dff,
                    x=f'{reduction_key_3d}_0',
                    y=f'{reduction_key_3d}_1',
                    z=f'{reduction_key_3d}_2',
                    color=dff[cluster_col].astype(str),
                    hover_data=hover_cols,
                    custom_data=custom,
                    title=f'{selected_algorithm}三维降维图',
                    color_discrete_sequence=CLUSTER_COLORS,
                    **symbol_kwargs_3d,
                )
                fig.update_traces(marker={'size': 1})

        if selected_dimension == 2:
            fig.update_traces(marker={'size': 8})
        fig.update_layout(uirevision='tsne-plot')

        params = data_store.get('params', {})
        if selected_algorithm == 'tsne':
            params[reduction_key] = {'perplexity': 30}
        elif selected_algorithm == 'umap':
            params[reduction_key] = {'n_neighbors': 15, 'min_dist': 0.1}

        updated_data_store = {
            'feature_cols': feature_cols,
            'raw_feature_cols': raw_feature_cols,
            'cluster_col': cluster_col,
            'image_col': image_col,
            'cluster_mode': data_cache.get('cluster_mode', 'merged'),
            'params': params
        }

        sample_cluster_mapping = df.set_index('sample_id')[cluster_col].to_dict()
        clusters = sorted(df[cluster_col].dropna().unique())
        cluster_options = [{'label': str(int(c)), 'value': int(c)} for c in clusters]

        return fig, updated_data_store, sample_cluster_mapping, cluster_options

    @app.callback(
        Output('cluster-images-store', 'data'),
        Output('last-selected-store', 'data'),
        Output('sample-panel', 'children'),
        Output('selected-meta', 'children'),
        Input('tsne-plot', 'clickData'),
        State('data-store', 'data')
    )
    def show_selected(clickData, data_store=None):
        if not clickData:
            return [], {}, html.Div('点击一个点以查看图片'), ''

        pts = clickData.get('points', [])
        if len(pts) == 0:
            return [], {}, html.Div('点击一个点以查看图片'), ''

        base_root = Path(__file__).parent.parent.parent
        image_root_abs = Path(image_root) if Path(image_root).is_absolute() else base_root / image_root

        p = pts[0]
        cd = p.get('customdata') or []
        sample_id = cd[0] if len(cd) >= 1 else None
        img_name = cd[1] if len(cd) >= 2 else p.get('hovertext')
        paired_images = cd[2] if len(cd) >= 3 else None

        data_cache = get_data_cache()
        df = data_cache['df']
        cluster_col = data_cache['cluster_col']
        image_col = data_cache['image_col']

        #df = ensure_sample_ids(df, image_col)

        row = None
        if sample_id is not None:
            row_candidates = df[df.get('sample_id') == sample_id]
            if len(row_candidates) > 0:
                row = row_candidates.iloc[0]

        if row is None and img_name is not None:
            row_candidates = df[df.get('image_name') == img_name]
            if len(row_candidates) > 0:
                row = row_candidates.iloc[0]

        if row is None:
            x = p.get('x')
            y = p.get('y')
            for algo in ['tsne', 'umap']:
                if f'{algo}_0' in df.columns and f'{algo}_1' in df.columns:
                    row_candidate = df[(np.isclose(df[f'{algo}_0'], x)) & (np.isclose(df[f'{algo}_1'], y))]
                    if len(row_candidate) > 0:
                        row = row_candidate.iloc[0]
                        break
        if row is None:
            return [], {}, html.Div('未找到对应记录'), ''

        sample_id = sample_id or row.get('sample_id')
        paired_images = paired_images or row.get('paired_images')

        if sample_id is None:
            name = str(row['image_name'])
            sample_id = Path(name).stem.replace('_exterior', '').replace('_interior', '')

        paired_names = []

        def side_label(name: str):
            low = str(name).lower()
            if 'interior' in low:
                return '内侧'
            if 'exterior' in low:
                return '外侧'
            return '未知侧'

        current_img = None
        if img_name:
            current_img = str(img_name)
            paired_names.append(current_img)
        elif row.get(image_col):
            current_img = str(row.get(image_col))
            paired_names.append(current_img)

        if len(paired_names) < 2:
            base_key = None
            if sample_id:
                base_key = str(sample_id).replace('_exterior', '').replace('_interior', '')
            elif img_name:
                base_key = Path(str(img_name)).stem
                base_key = base_key.replace('_exterior', '').replace('_interior', '')

            if base_key:
                candidates = []
                if 'image_name' in df.columns:
                    candidates = df[df['image_name'].str.contains(base_key, na=False)]['image_name'].tolist()
                if len(candidates) < 2 and 'sample_id' in df.columns:
                    candidates = candidates + df[df['sample_id'].str.contains(base_key, na=False)]['image_name'].tolist()

                candidates = list({Path(c).name for c in candidates})
                if len(candidates) > 1:
                    paired_names.extend(candidates)

        base_ext = Path(str(img_name)).suffix or '.png'

        def resolve_image_path(name, fallback_dir):
            name = Path(str(name)).name

            fb_dir = Path(fallback_dir)
            if not fb_dir.is_absolute():
                fb_dir = base_root / fb_dir

            def search_with_name(candidate_name: str):
                candidate = fb_dir / candidate_name
                if candidate.exists():
                    return candidate
                cluster_root = base_root / 'all_kmeans_new'
                for root, _, files in os.walk(cluster_root):
                    if candidate_name in files:
                        return Path(root) / candidate_name
                cutout_path = base_root / 'all_cutouts' / candidate_name
                if cutout_path.exists():
                    return cutout_path
                return None

            primary = search_with_name(name)
            if primary:
                return primary

            if Path(name).suffix == '':
                for ext in [base_ext, '.png', '.jpg', '.jpeg']:
                    alt = search_with_name(f"{name}{ext}")
                    if alt:
                        return alt
                if '_exterior' not in name and '_interior' not in name:
                    for side in ['_exterior', '_interior']:
                        for ext in [base_ext, '.png', '.jpg', '.jpeg']:
                            alt = search_with_name(f"{name}{side}{ext}")
                            if alt:
                                return alt
            return fb_dir / name

        base_dir = image_root_abs
        if 'image_path' in row and pd.notna(row['image_path']):
            base_dir = Path(row['image_path']).parent
        else:
            candidate = Path(str(row[image_col]))
            if candidate.parent != candidate:
                base_dir = candidate.parent
        print(f"[INFO] sample base_dir={base_dir}, image_root={image_root_abs}, img_name={img_name}")

        if len(paired_names) < 2 and current_img:
            stem = Path(current_img).stem
            suffix = Path(current_img).suffix or base_ext
            alt_candidates = []
            if 'interior' in stem:
                alt_candidates.append(stem.replace('interior', 'exterior') + suffix)
            elif 'exterior' in stem:
                alt_candidates.append(stem.replace('exterior', 'interior') + suffix)
            else:
                alt_candidates.append(f"{stem}_interior{suffix}")
                alt_candidates.append(f"{stem}_exterior{suffix}")

            for alt in alt_candidates:
                alt_name = Path(alt).name
                if alt_name in paired_names:
                    continue
                ipath = resolve_image_path(alt_name, base_dir)
                if ipath.exists():
                    paired_names.append(alt_name)

        seen = set()
        ordered = []
        for nm in paired_names:
            key = Path(str(nm)).name
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        paired_names = ordered
        sample_imgs = []
        for i, nm in enumerate(paired_names):
            ipath = resolve_image_path(nm, base_dir)
            if not ipath.exists():
                print(f"[WARN] sample image missing: {ipath} (nm={nm}, sample_id={sample_id})")
                sample_imgs.append(html.Div(f"缺少图片: {Path(nm).name}"))
                continue

            b64 = img_to_base64(ipath)
            b64_full = img_to_base64_full(ipath)
            if b64:
                side_txt = side_label(nm)
                sample_imgs.append(html.Div([
                    html.Img(
                        src=b64,
                        id=f'sample-img-{i}',
                        **{'data-image-path': str(Path(nm).name)},
                        **{'data-full-src': b64_full if b64_full else b64},
                        style={
                            'height': '200px',
                            'border': '1px solid #ccc',
                            'marginRight': '6px',
                            'cursor': 'pointer'
                        },
                        title='点击放大查看'
                    ),
                    html.Div(side_txt, style={'textAlign': 'center', 'fontSize': '12px', 'color': '#555', 'marginTop': '4px'})
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}))
            else:
                print(f"[WARN] failed to encode image: {ipath} (sample_id={sample_id})")
                sample_imgs.append(html.Div(f"图片无法读取: {Path(nm).name}"))

        if len(sample_imgs) == 0:
            sample_imgs = [html.Div('未找到正反面图片')]

        left_col = html.Div(sample_imgs, style={'display': 'flex', 'gap': '8px', 'alignItems': 'flex-start', 'flexWrap': 'wrap'})

        cluster_val = row[cluster_col]
        same_cluster = df[df[cluster_col] == cluster_val]
        image_paths = []
        for _, r in same_cluster.iterrows():
            base_dir_cluster = image_root_abs
            if 'image_path' in r and pd.notna(r['image_path']):
                base_dir_cluster = Path(r['image_path']).parent
            else:
                candidate = Path(str(r[image_col]))
                if candidate.parent != candidate:
                    base_dir_cluster = candidate.parent
            names = []
            if 'paired_images' in r and pd.notna(r['paired_images']):
                names = [n for n in str(r['paired_images']).split(';') if n]
            elif 'image_name' in r:
                names = [str(r['image_name'])]
            for nm in names:
                ipath = resolve_image_path(nm, base_dir_cluster)
                if not ipath.exists():
                    print(f"[WARN] cluster thumb missing: {ipath} (nm={nm}, cluster={cluster_val})")
                    continue
                image_paths.append(str(ipath))

        meta_parts = [
            html.B(row['image_name']),
            html.Br(),
            f"聚类: {cluster_val}",
            html.Br(),
            f"样本ID: {sample_id}"
        ]

        if 'sherd_id' in row.index and pd.notna(row['sherd_id']):
            meta_parts.extend([html.Br(), f"陶片ID: {row['sherd_id']}"])
        if 'unit_C' in row.index and pd.notna(row['unit_C']):
            meta_parts.extend([html.Br(), f"单位: {row['unit_C']}"])
        if 'part_C' in row.index and pd.notna(row['part_C']):
            meta_parts.extend([html.Br(), f"部位: {row['part_C']}"])
        if 'type_C' in row.index and pd.notna(row['type_C']):
            meta_parts.extend([html.Br(), f"类型: {row['type_C']}"])
        if 'part' in row.index and pd.notna(row['part']):
            meta_parts.extend([html.Br(), f"Part: {row['part']}"])
        if 'type' in row.index and pd.notna(row['type']):
            meta_parts.extend([html.Br(), f"Type: {row['type']}"])

        meta = html.Div(meta_parts, style={'fontSize': '13px', 'lineHeight': '1.6', 'color': '#333'})

        rep_name = None
        if 'image_name' in row:
            rep_name = row['image_name']
        elif paired_images:
            rep_name = paired_images[0]
        rep_path = None
        if rep_name:
            base_dir_sel = image_root_abs
            if 'image_path' in row and pd.notna(row['image_path']):
                base_dir_sel = Path(row['image_path']).parent
            else:
                candidate = Path(str(row[image_col]))
                if candidate.parent != candidate:
                    base_dir_sel = candidate.parent
            pth = resolve_image_path(rep_name, base_dir_sel)
            if not pth.exists():
                print(f"[WARN] compare path missing: {pth} for rep_name={rep_name}")
            rep_path = str(pth) if pth else None
        last_selected = {
            'sample_id': str(sample_id),
            'cluster': str(cluster_val),
            'name': str(rep_name) if rep_name else '未知图像',
            'path': rep_path or ''
        }

        sample_panel_children = html.Div([
            html.Div(left_col, style={'flex': '1 1 60%', 'minWidth': '280px'}),
            html.Div(meta, style={'flex': '1 1 40%', 'minWidth': '220px', 'paddingLeft': '8px'})
        ], style={'display': 'flex', 'gap': '12px', 'alignItems': 'flex-start'})

        return image_paths, last_selected, sample_panel_children, ''

    return app
