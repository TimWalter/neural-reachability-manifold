import torch


import ram.dataset.se3 as se3
from tqdm import tqdm
import pickle
from ram.dataset.loader import ValidationSet
from pathlib import Path

from paper_archive.rq2_accuracy.generative_graphik.generative_graphik.model import Model
from paper_archive.rq2_accuracy.adapter import network_args, forward_pose

torch.manual_seed(1)
device = torch.device("cuda")
batch_size = 1000
num_samples = 32

for path in ["test_boundary", "test"]:

    model = Model(network_args())
    model.load_state_dict(torch.load("/home/wtim/generative-graphik/saved_models/NRM/checkpoints/checkpoint.pth")["net"])
    model = model.to(device)

    graph = pickle.load(open(Path(__file__).parent / "data" / path / "graphs.pickle", "rb"))
    data = pickle.load(open(Path(__file__).parent / "data" / path / "data.pickle", "rb"))
    eval_set = ValidationSet(batch_size, False, path)

    se3_dist = []
    labels = []
    for batch_idx, (morph, pose, label) in enumerate(tqdm(eval_set, desc=f"Validation")):
        if path == "test" and batch_idx == 3000:
            break
        morph = morph.to(device, non_blocking=True)
        pose = se3.from_vector(pose.to(device, non_blocking=True))
        labels += [label]
        morph_idx = eval_set._get_batch(batch_idx)[:, 0].long()
        predicted_pose = forward_pose(model, data[batch_idx*batch_size: (batch_idx+1)*batch_size], num_samples,
                                 morph, pose, morph_idx, graph)

        se3_dist += [se3.distance(predicted_pose[:, -1], pose).cpu()]

    se3_dist = torch.cat(se3_dist, dim=0)
    labels = torch.cat(labels, dim=0)

    print(se3_dist[labels].mean())
    print(se3_dist[~labels].mean())

    torch.save(se3_dist, Path(__file__).parent / "data" / path / "se3_dist.pth")
    torch.save(labels, Path(__file__).parent / "data" / path / "labels.pth")
