import torch
import plotly.graph_objects as go
from paper_archive.utils import get_plotly_colour

torch.manual_seed(0)

voxels_per_side = 4
voxel_size = 1.0 / voxels_per_side
n = voxels_per_side
n_voxels = n ** 3

# All voxel origins: (n^3, 3)
idx = torch.arange(n)
gi, gj, gk = torch.meshgrid(idx, idx, idx, indexing='ij')
origins = torch.stack([gi.flatten(), gj.flatten(), gk.flatten()], dim=1).float() * voxel_size

# Vertex offsets and all vertices: (n_voxels, 8, 3)
offsets = torch.tensor([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
], dtype=torch.float32) * voxel_size
all_vertices = origins[:, None, :] + offsets[None, :, :]

faces = torch.tensor([
    [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [1, 2, 6], [1, 6, 5], [0, 3, 7], [0, 7, 4]
])

edge_pairs = torch.tensor([
    [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
    [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]
])

color_idx = torch.randint(0, 2, (n_voxels,))

# One Mesh3d trace per color (instead of one per voxel)
fig = go.Figure()
for c in range(2):
    mask = color_idx == c
    verts_c = all_vertices[mask]        # (m, 8, 3)
    m = verts_c.shape[0]
    verts_flat = verts_c.reshape(-1, 3)
    face_offsets = torch.arange(m)[:, None, None] * 8
    faces_flat = (faces[None].expand(m, -1, -1) + face_offsets).reshape(-1, 3)
    fig.add_trace(go.Mesh3d(
        x=verts_flat[:, 0], y=verts_flat[:, 1], z=verts_flat[:, 2],
        i=faces_flat[:, 0], j=faces_flat[:, 1], k=faces_flat[:, 2],
        color=get_plotly_colour(c), opacity=0.35, hoverinfo='skip',
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.5, fresnel=0.1),
        lightposition=dict(x=100, y=200, z=50)
    ))

# Hit points: batch-generate for all hit voxels, mask by num_points
hit_origins = origins[color_idx == 1]
m_hit = hit_origins.shape[0]
num_points = torch.randint(1, 25, (m_hit,))
max_pts = int(num_points.max().item())
rand_pts = torch.rand(m_hit, max_pts, 3) * voxel_size + hit_origins[:, None, :]
point_mask = torch.arange(max_pts)[None, :] < num_points[:, None]
hit_points = rand_pts[point_mask]

# Inner voxel edges: interleave start/end/NaN along each edge
edge_starts = all_vertices[:, edge_pairs[:, 0], :]  # (n_voxels, 12, 3)
edge_ends   = all_vertices[:, edge_pairs[:, 1], :]
nan_seg = torch.full_like(edge_starts, float('nan'))
edges_flat = torch.stack([edge_starts, edge_ends, nan_seg], dim=2).reshape(-1, 3)

fig.add_trace(go.Scatter3d(
    x=edges_flat[:, 0].tolist(), y=edges_flat[:, 1].tolist(), z=edges_flat[:, 2].tolist(),
    mode='lines', line=dict(color='black', width=1), hoverinfo='skip'
))

# Outer cube edges
outer_verts = torch.tensor([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
], dtype=torch.float32)
outer_starts = outer_verts[edge_pairs[:, 0]]
outer_ends   = outer_verts[edge_pairs[:, 1]]
nan_outer = torch.full_like(outer_starts, float('nan'))
outer_flat = torch.stack([outer_starts, outer_ends, nan_outer], dim=1).reshape(-1, 3)

fig.add_trace(go.Scatter3d(
    x=outer_flat[:, 0].tolist(), y=outer_flat[:, 1].tolist(), z=outer_flat[:, 2].tolist(),
    mode='lines', line=dict(color='black', width=2), hoverinfo='skip'
))

fig.add_trace(go.Scatter3d(
    x=hit_points[:, 0].tolist(), y=hit_points[:, 1].tolist(), z=hit_points[:, 2].tolist(),
    mode='markers',
    marker=dict(size=7, color='black', symbol='circle', opacity=1.0),
    hoverinfo='skip'
))

fig.update_layout(
    scene=dict(
        xaxis=dict(showbackground=False, showgrid=False, zeroline=False,
                   showticklabels=False, title_text='', range=[-0.05, 1.05]),
        yaxis=dict(showbackground=False, showgrid=False, zeroline=False,
                   showticklabels=False, title_text='', range=[-0.05, 1.05]),
        zaxis=dict(showbackground=False, showgrid=False, zeroline=False,
                   showticklabels=False, title_text='', range=[-0.05, 1.05]),
        bgcolor='rgba(0,0,0,0)',
        aspectmode='data'
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, b=0, t=40)
)
fig.update_layout(showlegend=False)

fig.write_image(
    "1_assign_labels_r3.png",
    width=2000,
    height=2000,
    scale=2
)

fig.show()
