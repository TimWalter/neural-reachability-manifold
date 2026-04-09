import argparse
from pathlib import Path

import torch
import zarr

import nrm.dataset.se3 as se3
from nrm.dataset.loader import ValidationSet
from nrm.dataset.boundaries import sample_boundary, generate_geodesic, generate_slice, generate_sphere

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="val", help="Base set")
args = parser.parse_args()

base_set = ValidationSet(1000, False, path=args.path)
compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

# Boundary poses
num_geodesics = 100
num_samples = 100
safe_folder = Path(__file__).parent.parent / 'data' / (args.path + "_boundary")
safe_folder.mkdir(parents=True, exist_ok=True)
root = zarr.open(safe_folder, mode="a")

morph_filename = "0_morphologies"
sample_filename = "0_samples"

root.create_array(
    morph_filename,
    shape=(num_geodesics, 8, 3),
    dtype="float32",
    chunks=(num_geodesics, 8, 3),
    compressors=compressor,
    overwrite=False,
)
root.create_array(sample_filename,
                  shape=(num_geodesics * num_samples, 11),
                  dtype="float32",
                  chunks=(num_geodesics * num_samples, 11),
                  shards=(num_geodesics * num_samples, 11),
                  compressors=compressor)

morphs, poses, labels = sample_boundary(base_set, num_geodesics, num_samples)
root[morph_filename] = morphs[::num_samples].numpy()
poses = se3.to_vector(poses)
labels = labels.float().unsqueeze(1)
morph_ids = torch.arange(0, num_geodesics).repeat_interleave(num_samples).unsqueeze(1)
root[sample_filename] = torch.cat([morph_ids, poses, labels], dim=1).numpy()

for fn, name, num_samples, exp in zip([generate_geodesic, generate_slice, generate_sphere],
                                      ["geodesic", "slice", "sphere"],
                                      [1000, 100, 100],
                                      [1, 2, 2]):
    safe_folder = Path(__file__).parent.parent / 'data' / (args.path + "_" + name)
    safe_folder.mkdir(parents=True, exist_ok=True)
    root = zarr.open(safe_folder, mode="a")

    morph_filename = "0_morphologies"
    sample_filename = "0_samples"

    root.create_array(
        morph_filename,
        shape=(1, 8, 3),
        dtype="float32",
        chunks=(1, 8, 3),
        compressors=compressor,
        overwrite=False,
    )
    root.create_array(sample_filename,
                      shape=(num_samples ** exp, 11),
                      dtype="float32",
                      chunks=(num_samples ** exp, 11),
                      shards=(num_samples ** exp, 11),
                      compressors=compressor)
    morphs, poses, labels = fn(base_set, num_samples)
    root[morph_filename] = morphs[0:1].numpy()
    poses = se3.to_vector(poses)
    labels = labels.float().unsqueeze(1)
    morph_ids = torch.zeros_like(labels)
    root[sample_filename] = torch.cat([morph_ids, poses, labels], dim=1).numpy()
