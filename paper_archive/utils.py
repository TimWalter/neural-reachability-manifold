import jax
import numpy as np
import jax.numpy as jnp
import optimistix as optx

from scipy.stats import bootstrap


def ci_95(data):
    if len(data) < 2:
        return (-1, -1)
    res = bootstrap((np.array(data),), np.mean, confidence_level=0.95, n_resamples=1000, method="basic")
    return res.confidence_interval.low.astype(np.float64), res.confidence_interval.high.astype(np.float64)


def bootstrap_mean_ci(trajectories, n_bootstraps=1000, ci=95):
    """
    Calculates the mean and confidence interval for a set of trajectories.

    Args:
        trajectories (np.ndarray): A 2D numpy array where each row is a trajectory.
                                   Shape: (n_trajectories, n_timepoints).
        n_bootstraps (int): The number of bootstrap samples to generate.
        ci (int): The desired confidence interval in percent.

    Returns:
        tuple: A tuple containing:
            - mean_trajectory (np.ndarray): The mean trajectory.
            - ci_lower (np.ndarray): The lower bound of the confidence interval.
            - ci_upper (np.ndarray): The upper bound of the confidence interval.
    """
    n_trajectories, n_timepoints = trajectories.shape
    bootstrap_means = np.zeros((n_bootstraps, n_timepoints))

    for i in range(n_bootstraps):
        indices = np.random.choice(n_trajectories, size=n_trajectories, replace=True)
        bootstrap_sample = trajectories[indices, :]
        bootstrap_means[i, :] = np.mean(bootstrap_sample, axis=0)

    mean_trajectory = np.mean(trajectories, axis=0)

    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    ci_lower, ci_upper = np.percentile(
        bootstrap_means, [lower_percentile, upper_percentile], axis=0
    )

    return mean_trajectory, ci_lower, ci_upper


def jax_distance(x1, x2):
    t1 = x1[..., :3, 3]
    r1 = x1[..., :3, :3]
    t2 = x2[..., :3, 3]
    r2 = x2[..., :3, :3]

    r_err = jnp.matmul(jnp.swapaxes(r1, -1, -2), r2)
    trace = r_err[..., 0, 0] + r_err[..., 1, 1] + r_err[..., 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = jnp.clip(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    rot_err = jnp.arccos(cos_angle)

    t_err_sq = jnp.sum((t1 - t2) ** 2, axis=-1, keepdims=True)
    total_dist_sq = t_err_sq / 8.0 + jnp.expand_dims(rot_err, -1) ** 2 / (2 * jnp.pi ** 2)

    return jnp.sqrt(total_dist_sq + 1e-12)


def jax_transformation_matrix(alpha, a, d, theta):
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    ct, st = jnp.cos(theta), jnp.sin(theta)
    zero = jnp.zeros_like(alpha)
    one = jnp.ones_like(alpha)

    row1 = jnp.concatenate([ct, -st, zero, a], axis=-1)
    row2 = jnp.concatenate([st * ca, ct * ca, -sa, -d * sa], axis=-1)
    row3 = jnp.concatenate([st * sa, ct * sa, ca, d * ca], axis=-1)
    row4 = jnp.concatenate([zero, zero, zero, one], axis=-1)
    return jnp.stack([row1, row2, row3, row4], axis=-2)


def jax_forward_kinematics(mdh, theta):
    transforms = jax_transformation_matrix(mdh[..., 0:1], mdh[..., 1:2], mdh[..., 2:3], theta)
    poses = []

    # Initialize base pose
    pose_shape = mdh.shape[:-2] + (4, 4)
    pose = jnp.broadcast_to(jnp.eye(4), pose_shape)

    for i in range(mdh.shape[-2]):
        pose = pose @ transforms[..., i, :, :]
        poses.append(pose)

    return jnp.stack(poses, axis=-3)


def ik_residual(joints, args):
    morph, target_pose = args
    reached_pose = jax_forward_kinematics(morph, joints)[-1]
    # TODO Proxy objective that is supposed to be smoother
    t_err = reached_pose[:3, 3] - target_pose[:3, 3]
    r_err = (reached_pose[:3, :3] - target_pose[:3, :3]).flatten()
    reg = joints.flatten() * 1e-4

    return jnp.concatenate([t_err, r_err, reg])


def solve_single_ik(morph, target_pose, init_joints):
    solver = optx.LevenbergMarquardt(rtol=1e-4, atol=1e-4)
    sol = optx.least_squares(
        fn=ik_residual,
        solver=solver,
        y0=init_joints,
        args=(morph, target_pose),
        max_steps=50,
        adjoint=optx.ImplicitAdjoint(),
        throw=False
    )
    return sol.value


jax_inverse_kinematics = jax.vmap(solve_single_ik, in_axes=(None, 0, 0))

def jax_from_vector(vec):
    translation = vec[..., :3]
    rotation_6d = vec[..., 3:]

    r1 = rotation_6d[..., :3]
    r2 = rotation_6d[..., 3:]
    r3 = jnp.cross(r1, r2, axis=-1)

    rot_matrix = jnp.stack([r1, r2, r3], axis=-1)
    top_block = jnp.concatenate([rot_matrix, translation[..., None]], axis=-1) # (..., 3, 4)

    batch_shape = vec.shape[:-1]
    bottom_row = jnp.array([0.0, 0.0, 0.0, 1.0])
    bottom_row = jnp.broadcast_to(bottom_row, (*batch_shape, 1, 4))

    homogeneous_matrix = jnp.concatenate([top_block, bottom_row], axis=-2)

    return homogeneous_matrix
