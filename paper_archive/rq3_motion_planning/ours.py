import torch

from tqdm import tqdm
from torch import Tensor
from jaxtyping import Float

import nrm.dataset.se3 as se3
import nrm.dataset.so3 as so3
import nrm.dataset.r3 as r3
from nrm.dataset.kinematics import numerical_inverse_kinematics
from nrm.model import MLP


def ours(morph: Float[Tensor, "dofp1 3"], target_trajectory: Float[Tensor, "num_samples 4 4"], n_iter: int,
         logging: bool = True) \
        -> tuple[
            list[float],  # Train Loss
            list[int],  # Reachability
            list[float],  # Deviation
            dict[str, list[float]]  # debug  # Predicted Reachability
        ]:
    bmorph = morph.unsqueeze(0).expand(target_trajectory.shape[0], -1, -1)

    position = target_trajectory[:, :3, 3].clone()
    orientation = target_trajectory[:, :3, :3].clone()

    position.requires_grad = True

    train_loss = []
    reachability = []
    deviation = []
    predicted_reachability = []

    delta_orientation = torch.zeros((target_trajectory.shape[0], 3), device=morph.device, requires_grad=True)
    optimizer = torch.optim.AdamW([position, delta_orientation], lr=0.002)
    model = MLP.from_id(13).to(morph.device)
    deviation_weights = torch.arange(0, target_trajectory.shape[0], device=target_trajectory.device).float()
    deviation_weights = deviation_weights.abs() /  deviation_weights.max()
    deviation_weights -= deviation_weights.mean()
    deviation_weights = deviation_weights.abs()**2
    print(deviation_weights)
    for _ in tqdm(range(n_iter)):
        optimizer.zero_grad()

        trajectory_vector = torch.cat([position, so3.to_vector(so3.exp(orientation, delta_orientation))], dim=-1)
        logit = model(bmorph, trajectory_vector)

        prediction_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(logit, torch.ones_like(logit))
        smoothness_loss = ((position[1:] - position[:-1]) ** 2).mean()
        deviation_loss = r3.distance(position, target_trajectory[:, :3, 3]).mean()
        loss = prediction_loss + 0.0 * smoothness_loss + 4.0 * deviation_loss
        loss.backward()

        optimizer.step()

        # Logging
        if logging:
            with torch.no_grad():
                train_loss += [loss.item()]
                trajectory = se3.from_vector(trajectory_vector)
                joints, manipulability = numerical_inverse_kinematics(morph, trajectory)
                reachability += [(manipulability != -1).sum()]
                deviation += [se3.distance(target_trajectory, trajectory).squeeze(-1).mean().item()]
                predicted_reachability += [torch.nn.Sigmoid()(logit).mean().item()]
    return train_loss, reachability, deviation, {"Predicted Reachability": predicted_reachability}, se3.from_vector(torch.cat([position, so3.to_vector(so3.exp(orientation, delta_orientation))], dim=-1))
