"""Modal component for image preview/controls."""
from dash import html, dcc


def build_modal():
    """Construct the image modal overlay and related hidden controls."""
    return html.Div([
        html.Div(id='image-modal', style={
            'display': 'none',
            'position': 'fixed',
            'top': 0,
            'left': 0,
            'width': '100%',
            'height': '100%',
            'backgroundColor': 'rgba(0, 0, 0, 0.9)',
            'zIndex': 9999,
            'cursor': 'pointer'
        }, children=[
            html.Div(style={
                'position': 'absolute',
                'top': '20px',
                'right': '20px',
                'zIndex': 10000,
                'display': 'flex',
                'gap': '10px'
            }, children=[
                html.Button('放大', id='zoom-in-btn', style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'border': 'none',
                    'padding': '8px 12px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '14px'
                }),
                html.Button('缩小', id='zoom-out-btn', style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'border': 'none',
                    'padding': '8px 12px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '14px'
                }),
                html.Button('左转', id='rotate-left-btn', style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'border': 'none',
                    'padding': '8px 12px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '14px'
                }),
                html.Button('右转', id='rotate-right-btn', style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'border': 'none',
                    'padding': '8px 12px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '14px'
                }),
                html.Button('重置', id='reset-btn', style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'border': 'none',
                    'padding': '8px 12px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '14px'
                }),
                html.Button('关闭', id='close-modal-btn', style={
                    'backgroundColor': 'rgba(255, 0, 0, 0.8)',
                    'color': 'white',
                    'border': 'none',
                    'padding': '8px 12px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '14px'
                })
            ]),
            html.Div(style={
                'position': 'absolute',
                'top': '0',
                'left': '0',
                'width': '100%',
                'height': '100%',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'overflow': 'visible'
            }, children=[
                html.Img(id='modal-image', style={
                    'maxWidth': '80vw',
                    'maxHeight': '80vh',
                    'border': '2px solid white',
                    'borderRadius': '4px',
                    'cursor': 'move',
                    'transition': 'transform 0.2s ease',
                    'transformOrigin': 'center center',
                    'position': 'relative',
                    'zIndex': 1
                }),
            ]),
            html.Div(style={
                'position': 'absolute',
                'bottom': '20px',
                'left': '50%',
                'transform': 'translateX(-50%)',
                'color': 'white',
                'fontSize': '14px',
                'textAlign': 'center'
            }, children=[
                html.Div('拖拽移动图片 | 滚轮缩放 | ESC键关闭', style={'opacity': '0.8'})
            ])
        ]),
        html.Div(id='image-click-trigger', style={'display': 'none'}),
        html.Div(id='modal-close-trigger', style={'display': 'none'}),
        dcc.Input(id='image-path-input', type='text', value='', style={'display': 'none'}),
        html.Script(src='/assets/image-modal.js'),
        html.Script(src='/assets/modal-inline.js')
    ], style={'margin': '8px', 'padding': '0'})
