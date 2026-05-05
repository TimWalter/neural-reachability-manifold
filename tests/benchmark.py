import torch
from tabulate import tabulate

import nrm.dataset.se3 as se3
from nrm.dataset.reachability_manifold import sample_poses_in_reach, estimate_reachability_manifold, sample_reachable_poses
from nrm.dataset.kinematics import inverse_kinematics
from nrm.dataset.morphology import sample_morph, get_joint_limits

from nrm.logger import binary_confusion_matrix

torch.manual_seed(1)

morphs = sample_morph(10, 5, False, torch.device("cpu"))

tp = []
fn = []
fp = []
tn = []
acc = []
f1 = []
minutes = []
reachable = []
benchmarks = []
for morph_idx, morph in enumerate(morphs):
    print(f"Morph_IDX: {morph_idx}")
    cell_indices = se3.index(sample_poses_in_reach(100_000, morph))
    _, manipulability = inverse_kinematics(morph, se3.cell(cell_indices.to(morph.device)))
    ground_truth = manipulability != -1
    reachable += [ground_truth.sum() / ground_truth.shape[0] * 100]

    if morph.shape[0] < 7:
        subsamples = 100_000 // 4
        bmorph = morph.unsqueeze(0).expand(subsamples, -1 ,-1)
        joint_limits = get_joint_limits(morph).unsqueeze(0).expand(subsamples, -1, -1)
        _, sub_cell_indices = sample_reachable_poses(bmorph, joint_limits)
        cell_indices[:sub_cell_indices.shape[0]] = sub_cell_indices.cpu()
        ground_truth[:subsamples] = True
    cell_indices = cell_indices.cpu()

    morph = morph.to("cuda")
    minutes += [0]
    true_positives = 0.0
    r_indices = torch.empty(0, dtype=torch.int64)
    while true_positives < 95.0 and minutes[-1] < 30:
        new_r_indices, benchmark, _ = estimate_reachability_manifold(morph, True, seconds=10)
        r_indices = torch.cat([r_indices, new_r_indices]).unique()
        benchmarks += [torch.tensor(benchmark)]
        minutes[-1] += 1/6

        labels = torch.isin(cell_indices, r_indices)

        (true_positives, false_negatives), (false_positives, true_negatives) = binary_confusion_matrix(labels,
                                                                                                       ground_truth)
        print(true_positives, false_negatives, false_positives, true_negatives)
        print(benchmarks[-1])
    tp += [true_positives]
    fn += [false_negatives]
    fp += [false_positives]
    tn += [true_negatives]
    acc += [(ground_truth == labels).sum() / labels.shape[0] * 100]
    f1 += [2 * true_positives / (2 * true_positives + false_positives + false_negatives) * 100]

mean_benchmark = torch.stack(benchmarks).mean(dim=0, keepdim=True).tolist()
mean_benchmark[0][0] = int(mean_benchmark[0][0])
mean_benchmark[0][1] = int(mean_benchmark[0][1])
print(tabulate(mean_benchmark,
               headers=["Filled Cells", "Total Samples<br>(Speed)", "Efficiency<br>(Total)", "Efficiency<br>(Unique)",
                        "Efficiency<br>(Collision)"], floatfmt=".4f", intfmt=",", tablefmt="github"))
print(tabulate(list(zip(tp, tn, fp, fn, f1, acc, reachable, minutes)),
               headers=["True Positives", "True Negatives", "False Positives", "False Negatives",
                        "F1 Score", "Accuracy", "Reachable", "Minutes"], floatfmt=".2f", tablefmt="github"))
