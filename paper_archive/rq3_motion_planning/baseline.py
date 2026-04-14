import jax
import torch
import optax
import jax.numpy as jnp

from tqdm import tqdm
from torch import Tensor
from jaxtyping import Float

import nrm.dataset.se3 as se3
import nrm.dataset.so3 as so3
from nrm.dataset.kinematics import numerical_inverse_kinematics

from paper_archive.utils import jax_inverse_kinematics, jax_forward_kinematics, jax_distance, jax_from_vector


def loss_fn(morph, position, orientation, init_joints):
    trajectory_vector = jnp.concatenate([position, orientation], axis=-1)
    trajectory = jax_from_vector(trajectory_vector)

    bmorph = jnp.broadcast_to(morph, (trajectory.shape[0], *morph.shape))
    optimal_joints = jax_inverse_kinematics(morph, trajectory, init_joints)

    reached_poses = jax.vmap(jax_forward_kinematics)(bmorph, optimal_joints)

    ee_poses = reached_poses[:, -1, :, :]
    dists = jax.vmap(jax_distance)(ee_poses, trajectory)

    loss = jnp.mean(dists)

    return loss, (trajectory_vector,)


loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=1)


def baseline(morph: Float[Tensor, "dofp1 3"], target_trajectory: Float[Tensor, "num_samples 4 4"], n_iter: int,
             logging: bool = True) \
        -> tuple[
            list[float],  # Train Loss
            list[int],  # Reachability
            list[float],  # Deviation
            dict[str, list[float]]  # debug
        ]:
    morph_torch = morph.clone()
    morph = jax.dlpack.from_dlpack(morph.contiguous().clone())
    position = jax.dlpack.from_dlpack(target_trajectory[:, :3, 3].contiguous().clone())
    orientation_vector = jax.dlpack.from_dlpack(so3.to_vector(target_trajectory[:, :3, :3]).contiguous().clone())

    train_loss = []
    reachability = []
    deviation = []

    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=0.01)
    )
    opt_state = optimizer.init(position)
    for i in tqdm(range(n_iter)):
        key = jax.random.PRNGKey(i)
        init_joints_jax = jax.random.uniform(key, shape=(position.shape[0], morph.shape[0], 1),
                                             minval=-jnp.pi, maxval=jnp.pi)

        (loss, (trajectory_vector,)), grads = loss_and_grad_fn(morph, position, orientation_vector, init_joints_jax)

        updates, opt_state = optimizer.update(grads, opt_state, position)
        position = optax.apply_updates(position, updates)

        # Logging
        if logging:
            with torch.no_grad():
                train_loss += [loss.item()]
                trajectory = se3.from_vector(torch.from_dlpack(trajectory_vector))
                joints, manipulability = numerical_inverse_kinematics(morph_torch, trajectory)
                reachability += [(manipulability != -1).sum()]
                deviation += [se3.distance(target_trajectory, trajectory).squeeze(-1).mean().item()]

    return train_loss, reachability, deviation, {}
