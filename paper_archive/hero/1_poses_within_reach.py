import nrm.dataset.r3 as r3
import nrm.dataset.se3 as se3
from nrm.dataset.reachability_manifold import estimate_reachable_ball
from nrm.dataset.kinematics import transformation_matrix
from nrm.dataset.morphology import sample_morph

import torch
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio

torch.manual_seed(0)
morph = sample_morph(1, 6, True)[0]

poses = se3.cell(torch.arange(se3.N_CELLS))
mask = poses[:, :3, 3].norm(dim=1) <= 1.0
poses = poses[mask]
centre, radius = estimate_reachable_ball(morph[:-1])
radius = max(0.0, radius - r3.DISTANCE_BETWEEN_CELLS)
eef_transformation = transformation_matrix(morph[-1, 0:1], morph[-1, 1:2], morph[-1, 2:3],
                                           torch.zeros_like(morph[-1, 0:1])).to(morph.device)
inverse = torch.linalg.inv(eef_transformation)
poses_without_eef = poses @ inverse.unsqueeze(0)
labels = poses_without_eef[:, :3, 3].norm(dim=1) < radius
positions = poses[:, :3, 3]

unique_positions, inverse_indices = torch.unique(positions, dim=0, return_inverse=True)

reach_count = torch.zeros(unique_positions.size(0), device=labels.device)
total_count = torch.zeros(unique_positions.size(0), device=labels.device)

# Scatter_add is efficient for this grouping operation
reach_count.scatter_add_(0, inverse_indices, labels.float())
total_count.scatter_add_(0, inverse_indices, torch.ones_like(labels.float()))

reachability_ratio = reach_count / total_count

torch.save(unique_positions, "unique_positions.pth")
torch.save(reachability_ratio, "reachability_ratio.pth")

unique_positions = torch.load("unique_positions.pth")
reachability_ratio = torch.load("reachability_ratio.pth")
# 1. Setup Colors
palette = sns.color_palette("colorblind")
c0 = np.array(palette[0])  # End color
c1 = np.array(palette[1])  # Start color (the vibrant one)

# A dark, matte charcoal for the shadow layer
# Must be standard RGBA decimal strings for plotly compatibility
dark_charcoal = 'rgba(26, 28, 29, 0.45)'

# 2. Prepare Data (Thinning must be preserved)
pos = unique_positions.cpu().numpy()
r_ratio = reachability_ratio.cpu().numpy()

# Randomized sampling breaks the grid.
indices = np.random.choice(len(pos), size=int(len(pos) * 0.05), replace=False)
pos_sub = pos[indices]
r_ratio_sub = r_ratio[indices]

# 3. Alpha for depth and contrast
opacities = (r_ratio_sub ** 1.8) * 0.8  # Aggressive alpha on edges
rgba_highlight = [
    f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a:.4f})'
    for (r, g, b), a in zip((c0 + (c1 - c0) * r_ratio_sub[:, None]), opacities)
]

# --- Layer 1: The "Shadow" (Base) ---
# We use the same positions, but override with dark color/low alpha.
trace_shadow = go.Scatter3d(
    x=pos_sub[:, 0], y=pos_sub[:, 1], z=pos_sub[:, 2],
    mode='markers',
    marker=dict(
        size=2.8,  # Slightly larger shadow than highlight
        color=dark_charcoal,  # List of dark strings
        line=dict(width=0)
    ),
    hoverinfo='skip'
)

# --- Layer 2: The "Highlight" (Top) ---
# The vibrant yellow/orange points go here.
trace_highlight = go.Scatter3d(
    x=pos_sub[:, 0], y=pos_sub[:, 1], z=pos_sub[:, 2],
    mode='markers',
    marker=dict(
        size=12,  # Optimized point size
        color=rgba_highlight,
        line=dict(width=0)
    ),
    hoverinfo='skip'
)

# 4. Build Figure (Shadow layer must come first!)
fig = go.Figure(data=[trace_shadow, trace_highlight])

# 5. Stylize Layout (Zero Padding preserved)
fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=0, pad=0),
    template="plotly_white",
    # CRITICAL: Keep it transparent
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    scene=dict(
        domain=dict(x=[0, 1], y=[0, 1]),  # Preserved max domain
        xaxis=dict(visible=False, showbackground=False),
        yaxis=dict(visible=False, showbackground=False),
        zaxis=dict(visible=False, showbackground=False),
        aspectmode='data',
        bgcolor='rgba(0,0,0,0)'  # Transparent scene
    ),
    # DYNAMIC CAMERA ANGLE (breaks grid feel)
    scene_camera=dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.1, y=1.3, z=0.55),  # Adjusted slightly closer and slightly less offset
        projection=dict(type='perspective')
    ),
    scene_aspectratio=dict(x=1, y=1, z=1)
)

# 6. High-Res Export (Zero padding preserved)
# If saving to PNG, ensure transparency is active.
pio.write_image(fig, "poses_in_reach.png", width=1100, height=1100, scale=3)
fig.show()
