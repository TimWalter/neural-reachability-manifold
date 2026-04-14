import torch
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

import nrm.dataset.se3 as se3

from nrm.dataset.morphology import sample_morph
from nrm.dataset.kinematics import numerical_inverse_kinematics, forward_kinematics
from nrm.dataset.self_collision import collision_check
from nrm.dataset.self_collision import LINK_RADIUS, EPS
from nrm.model import MLP

class Normaliser(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param):
        l2_norm = torch.hypot(param[:, 0:1], param[:, 1:2])
        norm = l2_norm.sum(dim=0, keepdim=True)
        ctx.save_for_backward(param, l2_norm, norm)
        return param / norm

    @staticmethod
    def backward(ctx, grad_output):
        param, l2_norm, norm = ctx.saved_tensors

        chain = torch.where(
            (param.abs() > EPS).any(dim=1, keepdim=True),
            param / l2_norm,
            torch.zeros_like(param)
        )

        return (grad_output * norm - chain * (grad_output * param).sum()) / norm ** 2


torch.manual_seed(1)
device = torch.device("cuda")

task = se3.random_ball(10, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.8])).to(device)
task_vec = se3.to_vector(task)

morph = sample_morph(1, 6, False, device)[0]

# Split morphology into discrete and continuous parameters
# 1. DISCRETE LOGITS (Gumbel parameters)
type_logits = torch.zeros(morph.shape[0], 4, device=device)
type_0 = (morph[:, 1] != 0) & (morph[:, 2] != 0)
type_logits[type_0, 0] = 1.0
type_1 = (morph[:, 1] == 0) & (morph[:, 2] != 0)
type_logits[type_1, 1] = 1.0
type_2 = (morph[:, 1] != 0) & (morph[:, 2] == 0)
type_logits[type_2, 2] = 1.0
type_3 = (morph[:, 1] == 0) & (morph[:, 2] == 0)
type_logits[type_3, 3] = 1.0

alpha_logits = torch.zeros(morph.shape[0], 3, device=device)
alpha_logits[morph[:, 0] == 0, 0] = 1.0
alpha_logits[morph[:, 0] == torch.pi/2, 1] = 1.0
alpha_logits[morph[:, 0] == -torch.pi/2, 2] = 1.0

sign_a_logits = torch.zeros(morph.shape[0], 2, device=device)
sign_a_logits[morph[:, 1] > 0, 0] = 1.0
sign_a_logits[morph[:, 1] < 0, 1] = 1.0
sign_a_logits[morph[:, 1] == 0, :] = 0.5

sign_d_logits = torch.zeros(morph.shape[0], 2, device=device)
sign_d_logits[morph[:, 2] > 0, 0] = 1.0
sign_d_logits[morph[:, 2] < 0, 1] = 1.0
sign_d_logits[morph[:, 2] == 0, :] = 0.5
# 2. CONTINUOUS PARAMETERS
length_offsets = morph[:, 1:] - 2 * LINK_RADIUS

type_logits.requires_grad = True
alpha_logits.requires_grad = True
sign_a_logits.requires_grad = True
sign_d_logits.requires_grad = True
length_offsets.requires_grad = True

train_loss = []
predicted_reachability = []
morphs = []
pose_error = []
self_collisions = []

optimizer = torch.optim.AdamW([type_logits, alpha_logits, sign_a_logits, sign_d_logits, length_offsets], lr=0.01)
tau = 1.0
tau_min = 0.05
decay_rate = 0.95
alpha_choices = torch.tensor([0.0, torch.pi / 2, -torch.pi / 2], device=device).unsqueeze(0)
model = MLP.from_id(13).to(device)
for i in tqdm(range(100)):
    optimizer.zero_grad()

    type_sample = torch.nn.functional.gumbel_softmax(type_logits, tau=tau, hard=False, dim=-1)
    alpha_sample = torch.nn.functional.gumbel_softmax(alpha_logits, tau=tau, hard=False, dim=-1)
    sign_a_sample = torch.nn.functional.gumbel_softmax(sign_a_logits, tau=tau, hard=False, dim=-1)
    sign_d_sample = torch.nn.functional.gumbel_softmax(sign_d_logits, tau=tau, hard=False, dim=-1)


    alpha = (alpha_sample * alpha_choices).sum(dim=-1, keepdim=True)

    sign_a = sign_a_sample[:, 0:1] - sign_a_sample[:, 1:2]
    sign_d = sign_d_sample[:, 0:1] - sign_d_sample[:, 1:2]
    mag_a = 2 * LINK_RADIUS + torch.nn.functional.softplus(length_offsets[:, 0:1])
    mag_d = 2 * LINK_RADIUS + torch.nn.functional.softplus(length_offsets[:, 1:2])

    a = sign_a * mag_a
    d = sign_d * mag_d

    type_0 = torch.cat([a, d], dim=1)
    type_1 = torch.cat([torch.zeros_like(a), d], dim=1)
    type_2 = torch.cat([a, torch.zeros_like(d)], dim=1)
    type_3 = torch.zeros_like(type_0)

    lengths = Normaliser.apply(
            type_sample[:, 0:1] * type_0 +
            type_sample[:, 1:2] * type_1 +
            type_sample[:, 2:3] * type_2 +
            type_sample[:, 3:4] * type_3
    )

    morph = torch.cat([alpha, lengths], dim=1)
    bmorph = morph.unsqueeze(0).expand(task.shape[0], -1, -1)
    logit = model(bmorph, task_vec)

    loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(logit, torch.ones_like(logit))

    loss.backward()
    optimizer.step()
    tau = max(tau_min, tau * decay_rate)

    with torch.no_grad():
        train_loss += [loss.item()]
        predicted_reachability += [torch.nn.Sigmoid()(logit).mean().item()]
        morphs += [morph.detach().clone().cpu()]
        joints = numerical_inverse_kinematics(morph, task)[0]
        reached_pose = forward_kinematics(bmorph, joints)
        pose_error += [se3.distance(reached_pose[:, -1, :, :], task).squeeze(-1).mean().item()]
        self_collisions += [collision_check(bmorph, reached_pose).sum().item()]


# Plot Training
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
colors = sns.color_palette("colorblind", 3)

ax1.plot(train_loss, label="Training loss", color=colors[0])
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.plot(predicted_reachability, label="Predicted Reachability", color=colors[1])
ax2.plot(pose_error, label="Pose Error", color=colors[2])
ax2.set_ylim(-0.1, 1.1)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

ax3.plot(self_collisions, label="Self Collisions", color=colors[1])
ax3.set_ylim(-0.1, task.shape[0]+0.1)
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Plot Morphs
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
colors = sns.color_palette("colorblind", morph.shape[0])
morphs = torch.stack(morphs, dim=0).cpu()

for i in range(morphs.shape[1]):
    ax1.plot(morphs[:, i, 0], label=rf"$\alpha_{i}$", color=colors[i], lw=1.5)
    ax2.plot(morphs[:, i, 1], label=rf"$a_{i}$", color=colors[i], lw=1.5)
    ax3.plot(morphs[:, i, 2], label=rf"$d_{i}$", color=colors[i], lw=1.5)

ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.axhline(2 * LINK_RADIUS, color='red', linestyle='--', alpha=0.3)
ax2.axhspan(EPS, 2 * LINK_RADIUS, color='red', alpha=0.15)
ax2.axhspan(-EPS, EPS, color='green', alpha=0.3)
ax2.axhspan(-2 * LINK_RADIUS, -EPS, color='red', alpha=0.15)
ax2.axhline(-2 * LINK_RADIUS, color='red', linestyle='--', alpha=0.3)

ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.axhline(2 * LINK_RADIUS, color='red', linestyle='--', alpha=0.3)
ax3.axhspan(EPS, 2 * LINK_RADIUS, color='red', alpha=0.15)
ax3.axhspan(-EPS, EPS, color='green', alpha=0.3)
ax3.axhspan(-2 * LINK_RADIUS, -EPS, color='red', alpha=0.15)
ax3.axhline(-2 * LINK_RADIUS, color='red', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
