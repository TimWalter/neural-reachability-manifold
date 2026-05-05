import numpy as np
import plotly.graph_objects as go

legend_colors = [f'rgba({int(92)}, {int(254)}, {int(254)}, {0.1:.4f})',
                 f'rgba({int(253)}, {int(251)}, {int(78)}, {0.7:.4f})',
                 f'rgba({int(255)}, {int(177)}, {int(6)}, {1:.4f})',]

# Tighten the spacing
legend_x = np.linspace(0, 0.32, 3)
legend_y = np.zeros(3)

# --- Trace 2: The Highlight (2D Circle) ---
trace_leg_highlight = go.Scatter(
    x=legend_x, y=legend_y,
    mode='markers',
    marker=dict(
        size=65, # Size in 2D pixels
        color=legend_colors,
        line=dict(width=0)
    ),
    hoverinfo='skip'
)

fig_leg = go.Figure(data=[trace_leg_highlight])

# 3. Layout for a Sleek Horizontal Legend
fig_leg.update_layout(
    width=1000, height=200,
    margin=dict(l=50, r=50, b=0, t=0), # Small side margins for safety
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    xaxis=dict(
        visible=False,
        range=[-0.2, 1.2],
        fixedrange=True # Prevents accidental zooming
    ),
    yaxis=dict(
        visible=False,
        range=[-0.5, 0.5],
        fixedrange=True,
        scaleanchor="x", # CRITICAL: Forces 1:1 aspect ratio for perfect circles
        scaleratio=1
    )
)

# High-Resolution Config for the Camera Icon
config = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'hero_legend_vector',
        'height': 500,
        'width': 2000,
        'scale': 4
    }
}

fig_leg.show(config=config)