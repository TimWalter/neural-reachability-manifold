import torch

from torch import Tensor
from jaxtyping import Float, Bool, jaxtyped
from beartype import beartype

import nrm.dataset.se3 as se3
from nrm.dataset.loader import ValidationSet
from nrm.dataset.kinematics import inverse_kinematics, transformation_matrix
from nrm.dataset.reachability_manifold import estimate_reachable_ball


@jaxtyped(typechecker=beartype)
def morph_and_endpoints(base_set: ValidationSet) \
        -> tuple[
            Float[Tensor, "1 dofp1 3"],
            Float[Tensor, "1 4 4"],
            Float[Tensor, "1 4 4"]
        ]:
    """
    Extract one morphology and one reachable&unreachable pose from the base_set.

    Args:
        base_set: Dataset to extract from.

    Returns:
        Morphology, unreachable pose, reachable pose.
    """
    morph = base_set.morphologies[torch.randint(0, len(base_set.morphologies), (1,)).item()].unsqueeze(0)
    dof = (morph[0].abs().sum(dim=1) != 0).sum().item()
    morph = morph[:, :dof]
    unreachable_pose = None
    reachable_pose = None
    for batch_idx, (comp_morph, pose, label) in enumerate(base_set):
        mask = (comp_morph == morph).all(dim=(1, 2))
        pose = pose[mask]
        label = label[mask]

        if (~label).any() and unreachable_pose is None:
            unreachable_pose = pose[~label][0:1]
        if label.any() and reachable_pose is None:
            reachable_pose = pose[label][0:1]
        if reachable_pose is not None and unreachable_pose is not None:
            break

    return morph, se3.from_vector(unreachable_pose), se3.from_vector(reachable_pose)

@jaxtyped(typechecker=beartype)
def generate_geodesic(base_set: ValidationSet, num_samples: int = 1000) \
        -> tuple[
            Float[Tensor, "num_samples dofp1 3"],
            Float[Tensor, "num_samples 4 4"],
            Bool[Tensor, "num_samples"]
        ]:
    """
    Given a base set, pick a morphology and compute a reachable and unreachable pose in their workspace.
    Sample the geodesic between these poses.

    Args:
         base_set: Dataset to extract from.
         num_samples: Number of samples to generate.

    Returns:
        Morphology, poses, labels
    """
    morph, start, end = morph_and_endpoints(base_set)

    tangent = se3.log(start, end)
    t = torch.linspace(0, 1, num_samples).view(-1, 1)
    poses = se3.exp(start.repeat(num_samples, 1, 1), t * tangent)
    labels = inverse_kinematics(morph[0].double(), poses.double())[1] != -1

    return morph.expand(num_samples, -1, -1).clone(), poses, labels

@jaxtyped(typechecker=beartype)
def sample_boundary(base_set: ValidationSet, num_geodesics: int, num_samples: int) \
        -> tuple[
            Float[Tensor, "{num_geodesics*num_samples} dofp1 3"],
            Float[Tensor, "{num_geodesics*num_samples} 4 4"],
            Bool[Tensor, "{num_geodesics*num_samples}"]
        ]:
    """
    Given a base set, compute boundary points by following geodesics across the boundary.

    Args:
        base_set: Base set providing morphologies and geodesic endpoints.
        num_geodesics: Number of geodesics to sample.
        num_samples: Number of samples per geodesic.

    Returns:
        Morphologies, poses, labels
    """
    morph_list = []
    pose_list = []
    label_list = []
    for i in range(num_geodesics):
        morph, poses, labels = generate_geodesic(base_set, num_samples)
        morph_list.append(morph)
        pose_list.append(poses)
        label_list.append(labels)
    return torch.cat(morph_list, dim=0), torch.cat(pose_list, dim=0), torch.cat(label_list, dim=0)

@jaxtyped(typechecker=beartype)
def morph_and_reachable(base_set: ValidationSet) \
        -> tuple[
            Float[Tensor, "1 dofp1 3"],
            Float[Tensor, "batch 4 4"]
        ]:
    """
    Extract one morphology and all its reachable poses.

    Args:
        base_set: Dataset to extract from.

    Returns:
        Morphology and poses.
    """
    morph = base_set.morphologies[torch.randint(0, len(base_set.morphologies), (1,)).item()].unsqueeze(0)
    dof = (morph[0].abs().sum(dim=1) != 0).sum().item()
    morph = morph[:, :dof]
    pose_list = []
    for batch_idx, (comp_morph, pose, label) in enumerate(base_set):
        mask = (comp_morph == morph).all(dim=(1, 2))
        pose = pose[mask]
        label = label[mask]

        pose_list += [pose[label]]
    poses = torch.cat(pose_list, dim=0)
    return morph, se3.from_vector(poses)

@jaxtyped(typechecker=beartype)
def generate_slice(base_set: ValidationSet, num_samples: int = 100) \
        -> tuple[
            Float[Tensor, "{num_samples*num_samples} dofp1 3"],
            Float[Tensor, "{num_samples*num_samples} 4 4"],
            Bool[Tensor, "{num_samples*num_samples}"]]:
    """
    Generate a 2D-slice through the workspace of a random morphology in the base set.
    The centre pose of the slice is picked as the pose with the median position. We fix the orientation and
    pick the slice normal to be the primary axes of rotation.

    Args:
        base_set: Dataset to extract from.
        num_samples: Number of samples in one direction.

    Returns:
        Morphology, poses, labels.
    """
    morph, poses = morph_and_reachable(base_set)
    centre, radius = estimate_reachable_ball(morph[0])

    mat = transformation_matrix(morph[:, 0, 0:1],
                                morph[:, 0, 1:2],
                                morph[:, 0, 2:3],
                                torch.zeros_like(morph[:, 0, 0:1]))
    torus_axis = torch.nn.functional.normalize(mat[0, :3, 2], dim=0)
    fixed_axes = torch.argmax(torus_axis.abs())
    axes_mask = torch.ones(3, dtype=torch.bool)
    axes_mask[fixed_axes] = False
    axes_range = torch.linspace(-radius, radius, num_samples)

    anchor = poses[torch.median(poses[:, :3, 3].norm(dim=1), dim=0).indices]
    poses = anchor.unsqueeze(0).expand(num_samples ** 2, -1, -1).clone()
    poses[:, :3, 3][:, axes_mask] = centre[axes_mask]

    poses[:, :3, 3][:, axes_mask] += torch.stack(torch.meshgrid(axes_range, axes_range, indexing='ij'),
                                                 dim=-1).reshape(-1, 2)
    labels = inverse_kinematics(morph[0].double(), poses.double())[1] != -1

    return morph.expand(num_samples ** 2, -1, -1).clone(), poses, labels

@jaxtyped(typechecker=beartype)
def generate_sphere(base_set: ValidationSet, num_samples: int = 100) \
        -> tuple[
            Float[Tensor, "{num_samples*num_samples} dofp1 3"],
            Float[Tensor, "{num_samples*num_samples} 4 4"],
            Bool[Tensor, "{num_samples*num_samples}"],
        ]:
    """
    Generate a sphere (S^2) through the workspace of a random morphology in the base set.
    The centre of the sphere is the end of the base link. The radius is picked as the median radius of reachable poses,
    and the orientation is the orientation of the pose whose radius has been picked.

    Args:
      base_set: Dataset to extract from.
      num_samples: Number of samples in one direction.

    Returns:
          Morphology, poses, labels.
    """
    morph, poses = morph_and_reachable(base_set)

    centre, radius = estimate_reachable_ball(morph[0])

    anchor = poses[torch.median(poses[:, :3, 3].norm(dim=1), dim=0).indices]
    poses = anchor.unsqueeze(0).expand(num_samples ** 2, -1, -1).clone()
    poses[:, :3, 3] = centre

    theta = torch.linspace(0, torch.pi, num_samples)
    phi = torch.linspace(0, 2 * torch.pi, num_samples)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

    x = anchor[:3, 3].norm() * torch.sin(theta_grid) * torch.cos(phi_grid)
    y = anchor[:3, 3].norm() * torch.sin(theta_grid) * torch.sin(phi_grid)
    z = anchor[:3, 3].norm() * torch.cos(theta_grid)

    poses[:, :3, 3] = centre + torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    labels = inverse_kinematics(morph[0].double(), poses.double())[1] != -1

    return morph[0].expand(num_samples ** 2, -1, -1).clone(), poses, labels
