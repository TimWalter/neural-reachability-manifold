import math
import torch
import plotly.graph_objects as go
from scipy.spatial import SphericalVoronoi

from paper_archive.utils import get_plotly_colour

def subdivide_triangle_on_sphere(a, b, c, edge_splits):
    """Subdivide one spherical triangle into smaller spherical triangles."""
    local_vertices = []
    vertex_index = {}

    for i in range(edge_splits + 1):
        for j in range(edge_splits + 1 - i):
            k = edge_splits - i - j
            p = (i * a + j * b + k * c) / edge_splits
            p /= torch.linalg.norm(p)
            vertex_index[(i, j)] = len(local_vertices)
            local_vertices.append(p)

    local_tris = []
    for i in range(edge_splits):
        for j in range(edge_splits - i):
            v0 = vertex_index[(i, j)]
            v1 = vertex_index[(i + 1, j)]
            v2 = vertex_index[(i, j + 1)]
            local_tris.append((v0, v1, v2))

            if j < edge_splits - i - 1:
                v3 = vertex_index[(i + 1, j + 1)]
                local_tris.append((v1, v3, v2))

    return torch.stack(local_vertices), local_tris


def build_spherical_cap(center, radius_angle, segments=20):
    """Build a small circular patch on the unit sphere centered at `center`."""
    n = center / torch.linalg.norm(center)

    helper = torch.tensor([0.0, 0.0, 1.0], dtype=n.dtype)
    if torch.dot(n, helper).abs() > 0.9:
        helper = torch.tensor([0.0, 1.0, 0.0], dtype=n.dtype)

    u = torch.linalg.cross(n, helper)
    u /= torch.linalg.norm(u)
    v = torch.linalg.cross(n, u)

    cap_vertices = [n]
    cap_i, cap_j, cap_k = [], [], []

    for seg in range(segments):
        theta0 = 2.0 * math.pi * seg / segments
        theta1 = 2.0 * math.pi * (seg + 1) / segments

        p0 = torch.cos(radius_angle) * n + torch.sin(radius_angle) * (
                math.cos(theta0) * u + math.sin(theta0) * v
        )
        p1 = torch.cos(radius_angle) * n + torch.sin(radius_angle) * (
                math.cos(theta1) * u + math.sin(theta1) * v
        )

        cap_vertices.append(p0 / torch.linalg.norm(p0))
        cap_vertices.append(p1 / torch.linalg.norm(p1))

        ring0 = 1 + 2 * seg
        ring1 = 2 + 2 * seg
        cap_i.append(0)
        cap_j.append(ring0)
        cap_k.append(ring1)

    return torch.stack(cap_vertices), cap_i, cap_j, cap_k


def sample_point_in_spherical_triangle(a, b, c):
    """Sample a point inside the spherical triangle using barycentric weights."""
    r1 = torch.rand(1).item()
    r2 = torch.rand(1).item()
    sqrt_r1 = r1 ** 0.5
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    p = u * a + v * b + w * c
    return p / torch.linalg.norm(p)

def visualize_full_geodesic_sphere(subdivisions=3, seed=None):
    """
    Creates a complete 3D visualization of a geodesic sphere, tiling the
    entire surface with its calculated hexagonal and pentagonal Voronoi cells.
    """

    if seed is not None:
        torch.manual_seed(seed)

    # --- 1. Generate Vertices by Subdividing an Icosahedron (same as before) ---
    t = (1.0 + 5.0 ** 0.5) / 2.0
    vertices = torch.tensor([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=torch.float64)
    vertices /= torch.linalg.norm(vertices[0])
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
        [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
        [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
        [8, 6, 7], [9, 8, 1]
    ]
    for _ in range(subdivisions):
        new_faces = []
        vertex_map = {}
        def get_midpoint(p1_idx, p2_idx, vertices, vertex_map):
            key = tuple(sorted((p1_idx, p2_idx)))
            if key not in vertex_map:
                mid = (vertices[p1_idx] + vertices[p2_idx]) / 2.0
                mid = mid / torch.linalg.norm(mid)
                vertices = torch.cat([vertices, mid.unsqueeze(0)], dim=0)
                vertex_map[key] = len(vertices) - 1
            return vertex_map[key], vertices
        for face in faces:
            m1, vertices = get_midpoint(face[0], face[1], vertices, vertex_map)
            m2, vertices = get_midpoint(face[1], face[2], vertices, vertex_map)
            m3, vertices = get_midpoint(face[2], face[0], vertices, vertex_map)
            new_faces.extend([[face[0], m1, m3], [face[1], m2, m1], [face[2], m3, m2], [m1, m2, m3]])
        faces = new_faces
    points = vertices.numpy()

    # --- 2. Calculate the Spherical Voronoi Diagram ---
    sv = SphericalVoronoi(points, 1, torch.tensor([0, 0, 0]))
    sv.sort_vertices_of_regions()

    # --- 3. Prepare Data for a Single, Efficient Mesh ---
    final_vertices = []
    all_tri_i, all_tri_j, all_tri_k = [], [], []
    all_face_colors = []
    edge_x, edge_y, edge_z = [], [], []
    marker_vertices = []
    marker_i, marker_j, marker_k = [], [], []
    marker_radius_angle = torch.tensor(0.012)
    marker_segments = 18

    # Loop through ALL Voronoi regions to build the mesh
    for region in sv.regions:
        poly_vertices = torch.tensor(sv.vertices[region], dtype=torch.float64)
        color_idx = torch.randint(0, 2, (1,)).item()
        current_color = get_plotly_colour(color_idx)

        # Triangulate each polygon face and refine into tiny spherical triangles.
        for j in range(len(poly_vertices) - 2):
            a = poly_vertices[0]
            b = poly_vertices[j + 1]
            c = poly_vertices[j + 2]
            local_vertices, local_tris = subdivide_triangle_on_sphere(a, b, c, 2)

            local_offset = len(final_vertices)
            final_vertices.extend(local_vertices.tolist())

            for tri in local_tris:
                all_tri_i.append(local_offset + tri[0])
                all_tri_j.append(local_offset + tri[1])
                all_tri_k.append(local_offset + tri[2])
                all_face_colors.append(current_color)

        # Add the polygon's edges to the wireframe list
        for j in range(len(region)):
            v_start = poly_vertices[j]
            v_end = poly_vertices[(j + 1) % len(region)]
            edge_x.extend([v_start[0].item(), v_end[0].item(), None])
            edge_y.extend([v_start[1].item(), v_end[1].item(), None])
            edge_z.extend([v_start[2].item(), v_end[2].item(), None])

        # In hit cells, add a few tiny circular black caps on the same sphere surface.
        if color_idx == 1:
            patch_count = torch.randint(0, 11, (1,)).item()
            if patch_count > 0:
                for _ in range(patch_count):
                    tri_idx = torch.randint(0, max(len(poly_vertices) - 2, 1), (1,)).item()
                    marker_center = sample_point_in_spherical_triangle(
                        poly_vertices[0],
                        poly_vertices[tri_idx + 1],
                        poly_vertices[tri_idx + 2],
                    )
                    cap_verts, cap_i, cap_j, cap_k = build_spherical_cap(
                        marker_center, marker_radius_angle, segments=marker_segments
                    )
                    marker_offset = len(marker_vertices)
                    marker_vertices.extend(cap_verts.tolist())
                    marker_i.extend([marker_offset + t for t in cap_i])
                    marker_j.extend([marker_offset + t for t in cap_j])
                    marker_k.extend([marker_offset + t for t in cap_k])

    # Consolidate all vertices into a single tensor
    final_vertices = torch.tensor(final_vertices)

    # --- 4. Create the Plotly Figure ---
    fig = go.Figure()

    # Add the single mesh for all colored faces
    fig.add_trace(go.Mesh3d(
        x=final_vertices[:, 0].tolist(), y=final_vertices[:, 1].tolist(), z=final_vertices[:, 2].tolist(),
        i=all_tri_i, j=all_tri_j, k=all_tri_k,
        facecolor=all_face_colors,  # Use facecolor to color each triangle
        opacity=1.0,
        hoverinfo='none',
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.5, roughness=0.4)
    ))

    if marker_vertices:
        marker_vertices = torch.tensor(marker_vertices)
        fig.add_trace(go.Mesh3d(
            x=marker_vertices[:, 0].tolist(), y=marker_vertices[:, 1].tolist(), z=marker_vertices[:, 2].tolist(),
            i=marker_i, j=marker_j, k=marker_k,
            color='black',
            opacity=1.0,
            hoverinfo='none',
            flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
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

    fig.write_image(
        "1_assign_labels_so3.png",
        width=2000,
        height=2000,
        scale=2
    )

    fig.show()

# --- Main execution ---
if __name__ == '__main__':
    visualize_full_geodesic_sphere(subdivisions=3)
