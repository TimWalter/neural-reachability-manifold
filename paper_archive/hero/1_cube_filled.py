import plotly.graph_objects as go
import numpy as np
import seaborn as sns

def visualize_voxel_cube_with_boundary(voxels_per_side=8, seed=None):
    """
    Creates a 3D visualization of a voxelized cube with a clear outer boundary,
    transparent inner voxels, and enhanced lighting.

    Args:
        voxels_per_side (int): The number of voxels along one side of the cube.
    """
    if not isinstance(voxels_per_side, int) or voxels_per_side < 1:
        raise ValueError("voxels_per_side must be a positive integer.")

    voxel_size = 1.0 / voxels_per_side

    # First two colors from seaborn's colorblind palette.
    palette = sns.color_palette("colorblind", 2)
    palette_rgb = [f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})" for r, g, b in palette]
    rng = np.random.default_rng(seed)

    # --- Data for Inner Voxel Outlines ---
    edge_x, edge_y, edge_z = [], [], []
    hit_points_x, hit_points_y, hit_points_z = [], [], []

    # --- Create the Plotly Figure ---
    fig = go.Figure()

    for i in range(voxels_per_side):
        for j in range(voxels_per_side):
            for k in range(voxels_per_side):
                x0, y0, z0 = i * voxel_size, j * voxel_size, k * voxel_size
                vertices = np.array([
                    [x0, y0, z0], [x0 + voxel_size, y0, z0],
                    [x0 + voxel_size, y0 + voxel_size, z0], [x0, y0 + voxel_size, z0],
                    [x0, y0, z0 + voxel_size], [x0 + voxel_size, y0, z0 + voxel_size],
                    [x0 + voxel_size, y0 + voxel_size, z0 + voxel_size], [x0, y0 + voxel_size, z0 + voxel_size]
                ])

                faces = np.array([
                    [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                    [1, 2, 6], [1, 6, 5], [0, 3, 7], [0, 7, 4]
                ])

                color_idx = int(rng.integers(0, 2))
                voxel_color = palette_rgb[color_idx]
                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color=voxel_color, opacity=0.35, hoverinfo='skip',
                    lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.5, fresnel=0.1),
                    lightposition=dict(x=100, y=200, z=50)
                ))

                # Treat the second palette color as a hit cell and place 0-10 random points inside it.
                if color_idx == 1:
                    num_points = int(rng.integers(1, 25))
                    if num_points > 0:
                        hit_points_x.extend(rng.uniform(x0, x0 + voxel_size, num_points))
                        hit_points_y.extend(rng.uniform(y0, y0 + voxel_size, num_points))
                        hit_points_z.extend(rng.uniform(z0, z0 + voxel_size, num_points))

                edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), 
                         (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
                for v_start, v_end in edges:
                    edge_x.extend([vertices[v_start][0], vertices[v_end][0], None])
                    edge_y.extend([vertices[v_start][1], vertices[v_end][1], None])
                    edge_z.extend([vertices[v_start][2], vertices[v_end][2], None])

    # --- NEW: Data for the Outer Cube Boundary ---
    outer_verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    outer_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), 
                   (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    outer_edge_x, outer_edge_y, outer_edge_z = [], [], []
    for v_start, v_end in outer_edges:
        outer_edge_x.extend([outer_verts[v_start][0], outer_verts[v_end][0], None])
        outer_edge_y.extend([outer_verts[v_start][1], outer_verts[v_end][1], None])
        outer_edge_z.extend([outer_verts[v_start][2], outer_verts[v_end][2], None])

    # 1. Add the inner voxel outlines
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode='lines',
        line=dict(color='black', width=1), hoverinfo='skip'
    ))

    # 2. Add the bold outer boundary
    fig.add_trace(go.Scatter3d(
        x=outer_edge_x, y=outer_edge_y, z=outer_edge_z, mode='lines',
        line=dict(color='black', width=2), # Thicker line for emphasis
        hoverinfo='skip'
    ))

    # 3. Add random points within hit cells
    fig.add_trace(go.Scatter3d(
        x=hit_points_x, y=hit_points_y, z=hit_points_z,
        mode='markers',
        marker=dict(size=7, color='black', symbol='circle', opacity=1.0),
        hoverinfo='skip'
    ))

    # --- Configure the Layout ---
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

    # --- Export the high-resolution PNG ---
    output_filename = "cube_filled.png"
    print(f"Saving high-resolution image to {output_filename}...")
    
    # Export a 2000x2000 image at 2x scale, resulting in a 4000x4000 pixel PNG
    fig.write_image(
        output_filename,
        width=2000,
        height=2000,
        scale=2
    )

    fig.show()


# --- Main execution ---
if __name__ == '__main__':
    resolution = 4
    visualize_voxel_cube_with_boundary(resolution)