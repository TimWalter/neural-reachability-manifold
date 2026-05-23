import torch

from ram.dataset.morphology import sample_morph, _reject_morph, get_joint_limits
from ram.dataset.kinematics import forward_kinematics, morph_to_eaik, is_analytically_solvable
from ram.dataset.self_collision import collision_check

torch.set_default_dtype(torch.float64)


def test_reject_degenerate_consecutive_parallel_axes():
    n_robots = 500
    morphs = sample_morph(n_robots, 6, False)

    axes_choice = torch.randint(1, 3, (n_robots,))
    row_indices = torch.arange(n_robots)

    morphs[row_indices, axes_choice, 0] = 0.0
    morphs[row_indices, axes_choice + 1, 0] = 0.0
    morphs[row_indices, axes_choice + 2, 0] = 0.0

    degenerate = _reject_morph(morphs)

    assert degenerate.all(), f"{(~degenerate).sum()} {morphs[~degenerate][0]}"


def test_reject_degenerate_collinear_axes():
    n_robots = 500
    morphs = sample_morph(n_robots, 6, False)

    axes_choice = torch.randint(0, 4, (n_robots,))
    row_indices = torch.arange(n_robots)

    morphs[row_indices, axes_choice, 1:] = torch.tensor([0, 0.1])
    morphs[row_indices, axes_choice + 1, :] = torch.tensor([0, 0, 0.1])

    degenerate = _reject_morph(morphs)
    assert degenerate.all(), f"{(~degenerate).sum()} {morphs[~degenerate][0]}"


def test_analytically_solvable_5dof():
    n_robots = 500
    morphs = sample_morph(n_robots, 5, True)
    for morph in morphs:
        eaik = morph_to_eaik(morph)
        assert eaik.hasKnownDecomposition(), morph


def test_analytically_solvable_6dof():
    n_robots = 500
    morphs = sample_morph(n_robots, 6, True)
    for i, morph in enumerate(morphs):
        eaik = morph_to_eaik(morph)
        assert eaik.hasKnownDecomposition(), f"{i}, {morph}"


def test_is_analytically_solvable_5dof():
    n_robots = 500
    morphs = sample_morph(n_robots, 5, True)
    assert is_analytically_solvable(morphs).all(), morphs[~is_analytically_solvable(morphs)]
    morphs = sample_morph(n_robots, 5, False)
    assert not is_analytically_solvable(morphs).all()


def test_is_analytically_solvable_6dof():
    n_robots = 10
    morphs = sample_morph(n_robots, 6, True)
    assert is_analytically_solvable(morphs).all(), morphs[~is_analytically_solvable(morphs)]
    morphs = sample_morph(n_robots, 6, False)
    assert not is_analytically_solvable(morphs).all()


def test_irrelevant_size():
    n_robots = 500
    morphs = sample_morph(n_robots, 6, False)

    size_factor = torch.rand(n_robots, 1, 1).expand(-1, 7, 2) * 10
    sized_morph = torch.cat((morphs[..., 0:1], size_factor * morphs[..., 1:]), dim=-1)

    joints = 2 * torch.pi * torch.rand(100, *morphs.shape[:-1], 1) - torch.pi
    joints[:, -1, :] = 0

    morph = morphs.unsqueeze(0).expand(100, -1, -1, -1)
    eef_poses = forward_kinematics(morph, joints)[..., -1, :, :]

    sized_morph = sized_morph.unsqueeze(0).expand(100, -1, -1, -1)
    sized_eef_poses = forward_kinematics(sized_morph, joints)[..., -1, :, :]

    orientations = eef_poses[..., :3, :3]
    sized_orientations = sized_eef_poses[..., :3, :3]
    assert (orientations == sized_orientations).all()
    positions = eef_poses[..., :3, 3]
    sized_positions = sized_eef_poses[..., :3, 3]
    size_factor = size_factor[:, 0, 0].unsqueeze(0).unsqueeze(-1).expand(100, -1, 3)
    assert (size_factor * positions - sized_positions).max() < 1e-6


def test_scissor_collision():
    normal_morphs = sample_morph(100, 6, False)
    analytical_morphs = sample_morph(100, 6, True)
    for morphs in [normal_morphs, analytical_morphs]:
        print("Morphs Type")
        for morph_idx, morph in enumerate(morphs):
            joint_limits = get_joint_limits(morph)

            extended_morph = torch.cat([torch.zeros_like(morph[:1]), morph])
            alpha0, a0, d0 = extended_morph[:-2].split(1, dim=-1)
            alpha1, a1, d1 = extended_morph[1:-1].split(1, dim=-1)
            alpha2, a2, d2 = extended_morph[2:].split(1, dim=-1)
            wrist = (a1[:, 0] == 0) & (d1[:, 0] == 0)
            limited = joint_limits[:-1, 0] != 2 * torch.pi
            for joint_idx in range(morph.shape[0] - 1):
                if not limited[joint_idx]:
                    continue
                isolated_morph = morph[joint_idx - (1 if wrist[joint_idx] else 0):joint_idx + 2, :].clone()
                if a2[joint_idx] != 0:
                    isolated_morph[-1, 2] = 0
                if wrist[joint_idx]:
                    if d0[joint_idx] != 0:
                        isolated_morph[0, 1] = 0
                else:
                    if d1[joint_idx] != 0:
                        isolated_morph[0, 1] = 0

                isolated_morph = isolated_morph.unsqueeze(0).expand(100, -1, -1)
                non_colliding_joints = torch.rand(100, isolated_morph.shape[1], 1,
                                                  device=morph.device) * joint_limits[joint_idx][0:1] + joint_limits[joint_idx][1:2]

                poses = forward_kinematics(isolated_morph, non_colliding_joints)
                critical_distance = collision_check(isolated_morph, poses, debug=True)
                assert (critical_distance >= 0.0).all(), \
                    f"{isolated_morph[0]} \n {non_colliding_joints[torch.argmin(critical_distance)]}"

                colliding_joints = torch.zeros(2, isolated_morph.shape[1], 1)
                colliding_joints[0, :] = 1.0
                colliding_joints = colliding_joints * joint_limits[joint_idx, 0:1] + joint_limits[joint_idx, 1:2]
                over_edge = torch.zeros_like(colliding_joints)
                over_edge[0, :] = torch.pi / 20
                over_edge[1, :] = - torch.pi / 20
                colliding_joints += over_edge
                isolated_morph = isolated_morph[0].unsqueeze(0).expand(2, -1, -1)
                poses = forward_kinematics(isolated_morph, colliding_joints)
                critical_distance = collision_check(isolated_morph, poses, debug=True)
                assert (critical_distance < 0.0).all(), f"{isolated_morph[0]} \n {colliding_joints[torch.argmax(critical_distance)]}"

