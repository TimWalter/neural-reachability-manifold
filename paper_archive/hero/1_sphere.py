import plotly.graph_objects as go
import numpy as np
from scipy.spatial import SphericalVoronoi

def visualize_full_geodesic_sphere(subdivisions=3):
    """
    Creates a complete 3D visualization of a geodesic sphere, tiling the
    entire surface with its calculated hexagonal and pentagonal Voronoi cells.
    """

    # --- 1. Generate Vertices by Subdividing an Icosahedron (same as before) ---
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ])
    vertices /= np.linalg.norm(vertices[0])
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
        [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
        [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
        [8, 6, 7], [9, 8, 1]
    ])
    for _ in range(subdivisions):
        new_faces = []
        vertex_map = {}
        def get_midpoint(p1_idx, p2_idx, vertices, vertex_map):
            key = tuple(sorted((p1_idx, p2_idx)))
            if key not in vertex_map:
                mid = (vertices[p1_idx] + vertices[p2_idx]) / 2.0
                mid /= np.linalg.norm(mid)
                vertices = np.vstack([vertices, mid])
                vertex_map[key] = len(vertices) - 1
            return vertex_map[key], vertices
        for face in faces:
            m1, vertices = get_midpoint(face[0], face[1], vertices, vertex_map)
            m2, vertices = get_midpoint(face[1], face[2], vertices, vertex_map)
            m3, vertices = get_midpoint(face[2], face[0], vertices, vertex_map)
            new_faces.extend([[face[0], m1, m3], [face[1], m2, m1], [face[2], m3, m2], [m1, m2, m3]])
        faces = np.array(new_faces)
    points = vertices

    # --- 2. Calculate the Spherical Voronoi Diagram ---
    sv = SphericalVoronoi(points, 1, np.array([0, 0, 0]))
    sv.sort_vertices_of_regions()

    # --- 3. Prepare Data for a Single, Efficient Mesh ---
    all_poly_vertices = []
    all_tri_i, all_tri_j, all_tri_k = [], [], []
    all_face_colors = []
    edge_x, edge_y, edge_z = [], [], []
    vertex_offset = 0
    
    # Define a color palette to cycle through
    colors = ['#FFFFFF']

    # Loop through ALL Voronoi regions to build the mesh
    for i, region in enumerate(sv.regions):
        poly_vertices = sv.vertices[region]
        current_color = colors[i % len(colors)]
        
        # Triangulate the polygon (fan triangulation)
        for j in range(len(poly_vertices) - 2):
            all_tri_i.append(vertex_offset)
            all_tri_j.append(vertex_offset + j + 1)
            all_tri_k.append(vertex_offset + j + 2)
            all_face_colors.append(current_color)
        
        # Add the polygon's vertices to our master list
        all_poly_vertices.append(poly_vertices)
        
        # Add the polygon's edges to the wireframe list
        for j in range(len(region)):
            v_start = poly_vertices[j]
            v_end = poly_vertices[(j + 1) % len(region)] # Wrap around
            edge_x.extend([v_start[0], v_end[0], None])
            edge_y.extend([v_start[1], v_end[1], None])
            edge_z.extend([v_start[2], v_end[2], None])

        vertex_offset += len(poly_vertices)

    # Consolidate all vertices into a single NumPy array
    final_vertices = np.vstack(all_poly_vertices)

    # --- 4. Create the Plotly Figure ---
    fig = go.Figure()

    # Add the single mesh for all colored faces
    fig.add_trace(go.Mesh3d(
        x=final_vertices[:, 0], y=final_vertices[:, 1], z=final_vertices[:, 2],
        i=all_tri_i, j=all_tri_j, k=all_tri_k,
        facecolor=all_face_colors,  # Use facecolor to color each triangle
        opacity=1.0,
        hoverinfo='none',
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.5, roughness=0.4)
    ))

    # Add the wireframe for all edges
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='black', width=3),
        hoverinfo='none'
    ))

    # --- 5. Configure Layout ---
    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)',
            aspectmode='data'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=40)
    )

# --- Export the high-resolution PNG ---
    output_filename = "sphere.png"
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
    visualize_full_geodesic_sphere(subdivisions=3)