import plotly.graph_objects as go
import numpy as np

def visualize_voxel_cube_with_boundary(voxels_per_side=8):
    """
    Creates a 3D visualization of a voxelized cube with a clear outer boundary,
    transparent inner voxels, and enhanced lighting.

    Args:
        voxels_per_side (int): The number of voxels along one side of the cube.
    """
    if not isinstance(voxels_per_side, int) or voxels_per_side < 1:
        raise ValueError("voxels_per_side must be a positive integer.")

    voxel_size = 1.0 / voxels_per_side
    
    # --- Data for Inner Voxels (as before) ---
    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
    edge_x, edge_y, edge_z = [], [], []
    vertex_offset = 0

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
                
                all_x.extend(vertices[:, 0])
                all_y.extend(vertices[:, 1])
                all_z.extend(vertices[:, 2])

                faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], 
                                  [0, 5, 4], [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], 
                                  [0, 3, 7], [0, 7, 4]]) + vertex_offset
                
                all_i.extend(faces[:, 0])
                all_j.extend(faces[:, 1])
                all_k.extend(faces[:, 2])

                edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), 
                         (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
                for v_start, v_end in edges:
                    edge_x.extend([vertices[v_start][0], vertices[v_end][0], None])
                    edge_y.extend([vertices[v_start][1], vertices[v_end][1], None])
                    edge_z.extend([vertices[v_start][2], vertices[v_end][2], None])

                vertex_offset += 8

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

    # --- Create the Plotly Figure ---
    fig = go.Figure()

    # 1. Add the transparent voxel faces
    fig.add_trace(go.Mesh3d(
        x=all_x, y=all_y, z=all_z, i=all_i, j=all_j, k=all_k,
        color='lightgrey', opacity=0.3, hoverinfo='skip',
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.5, fresnel=0.1),
        lightposition=dict(x=100, y=200, z=50)
    ))

    # 2. Add the inner voxel outlines
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode='lines',
        line=dict(color='black', width=1), hoverinfo='skip'
    ))

    # 3. NEW: Add the bold outer boundary
    fig.add_trace(go.Scatter3d(
        x=outer_edge_x, y=outer_edge_y, z=outer_edge_z, mode='lines',
        line=dict(color='black', width=2), # Thicker line for emphasis
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
    output_filename = "cube.png"
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