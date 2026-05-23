import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from liegroups.numpy import SE3
from graphik.robots import RobotRevolute
from graphik.graphs.graph_revolute import ProblemGraphRevolute
from paper_archive.rq2_accuracy.generative_graphik.generative_graphik.utils.dataset_generation import generate_struct_data, generate_data_point_from_pose

import ram.dataset.se3 as se3
from ram.dataset.loader import ValidationSet


for path in ["test_boundary", "test"]:
    eval_set = ValidationSet(1, False, path)

    robots = []
    graphs = []
    struct_data = []
    for morph_idx in tqdm(range(len(eval_set.morphologies)), "To graph"):
        morph = eval_set._get_morph(torch.tensor([morph_idx]))[0]

        params = {
            "alpha": morph[:, 0].tolist(),
            "a": morph[:, 1].tolist(),
            "d": morph[:, 2].tolist(),
            "theta": [0]*morph.shape[0],
            "num_joints": morph.shape[0],
            "modified_dh": True,
        }

        robots += [RobotRevolute(params)]
        graphs += [ProblemGraphRevolute(robots[-1])]
        struct_data += [generate_struct_data(graphs[-1])]

    data = []
    for batch_idx, (morph, pose, label) in enumerate(tqdm(eval_set, desc=path)):
        morph_idx = eval_set._get_batch(batch_idx)[0, 0].long()
        data += [generate_data_point_from_pose(graphs[morph_idx], SE3.from_matrix(se3.from_vector(pose[0]).numpy(), normalize=True), struct_data[morph_idx])]

    directory = Path(__file__).parent / "data" / path
    directory.mkdir(parents=True, exist_ok=True)
    pickle.dump(robots, open(directory / "robots.pickle", "wb"))
    pickle.dump(graphs, open(directory / "graphs.pickle", "wb"))
    pickle.dump(struct_data, open(directory / "struct_data.pickle", "wb"))
    pickle.dump(data, open(directory / "data.pickle", "wb"))

