import torch
from graphik.robots import RobotRevolute
from graphik.utils.dgp import graph_from_pos
from graphik.graphs.graph_revolute import ProblemGraphRevolute

import ram.dataset.se3 as se3
from ram.dataset.morphology import get_joint_limits, sample_morph
from ram.dataset.kinematics import forward_kinematics
from liegroups.numpy import SE3

from paper_archive.rq2_accuracy.generative_graphik.generative_graphik.utils.torch_utils import batchPmultiDOF

dof = 6
device = torch.device("cpu")
num_robots = 100
morph = sample_morph(num_robots, dof, False, device)
joint_limits = get_joint_limits(morph)
target_joints = torch.rand(*joint_limits.shape[:-1], 1, device=device) * joint_limits[..., 0:1] + joint_limits[..., 1:2]
pose_nrm = forward_kinematics(morph, target_joints)

graph_list = []
for m in morph:
    params = {
        "alpha": m[:, 0].tolist(),
        "a": m[:, 1].tolist(),
        "d": m[:, 2].tolist(),
        "theta": [0] * (dof+1), # GraphIK wants the rest pose here
        "num_joints": dof+1,
        "modified_dh": True,
    }
    robot = RobotRevolute(params)
    graph_list += [ProblemGraphRevolute(robot)]

identity = torch.eye(4, device=morph.device, dtype=morph.dtype).expand(*morph.shape[:-2], 1, 4, 4).clone()
extended_pose = torch.cat([identity, pose_nrm],dim=-3)
P_graphik = batchPmultiDOF(extended_pose.reshape(-1, 4, 4), torch.tensor([dof+1]*num_robots))
P_graphik  = P_graphik.reshape(num_robots, -1, 3)
joint_list = []
for graph, output, pose in zip(graph_list, P_graphik, pose_nrm):
    g_pos = graph_from_pos(output, graph.node_ids)
    joint_list += [torch.tensor(list(graph.joint_variables(g_pos, T_final={f"p{dof+1}": SE3.from_matrix(pose[-1].numpy(), normalize=True)}).values())).unsqueeze(0)]

joints_ggik_raw = torch.cat(joint_list, dim=0).unsqueeze(-1)
joints_ggik = torch.cat([joints_ggik_raw[:, 1:], torch.zeros_like(joints_ggik_raw[:, 0:1])], dim=1).float()

pose_ggik = forward_kinematics(morph, joints_ggik)

print(se3.distance(pose_ggik, pose_nrm).mean())
print(torch.min(se3.distance(pose_ggik, pose_nrm).mean(dim=1)))
print(torch.max(se3.distance(pose_ggik, pose_nrm).mean(dim=1)))
print(morph[torch.argmin(se3.distance(pose_ggik, pose_nrm).mean(dim=1))])
print(morph[torch.argmax(se3.distance(pose_ggik, pose_nrm).mean(dim=1))])
print(se3.distance(pose_ggik, pose_nrm)[torch.argmax(se3.distance(pose_ggik, pose_nrm).mean(dim=1))])
idx = torch.argmax(se3.distance(pose_ggik, pose_nrm).mean(dim=1))
print(joints_ggik_raw[idx])
print(target_joints[idx])
# It cannot correctly predict where the rotation shall go in case of coaxial axes (type1 + alpha=0)