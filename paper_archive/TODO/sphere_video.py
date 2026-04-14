import wandb
import cv2
import os
import shutil
import json
import numpy as np
import plotly.io as pio
from pathlib import Path
from tqdm import tqdm

def create_sphere_orbital_video(entity, project, run_id, output_name="sphere_rotation.mp4", fps=10, rotations=1):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    temp_dir = Path("temp_sphere_frames")
    temp_dir.mkdir(exist_ok=True)

    # Key from your logger.py: "Boundaries/Sphere"
    print(f"Fetching history for run: {run.name}")
    history = list(run.scan_history(keys=["Boundaries/Sphere", "_step"]))

    total_frames = len(history)
    processed_frames = []

    # Camera settings
    radius = 2.0  # Distance of camera from center
    z_height = 1.0 # Height of camera

    for i, row in enumerate(tqdm(history, desc="Rendering rotating frames")):
        if "Boundaries/Sphere" in row:
            step = row["_step"]
            image_meta = row["Boundaries/Sphere"]

            # Download Plotly JSON
            file_path = image_meta["path"]
            downloaded_file = run.file(file_path).download(root=temp_dir, replace=True)

            try:
                with open(downloaded_file.name, 'r') as f:
                    fig_dict = json.load(f)

                fig = pio.from_json(json.dumps(fig_dict))

                # Calculate camera angle (theta) based on frame index
                # This ensures synchronization across all subplots
                theta = (i / total_frames) * (2 * np.pi * rotations)
                eye_x = radius * np.cos(theta)
                eye_y = radius * np.sin(theta)

                # Update all scenes (subplots) simultaneously
                fig.update_scenes(
                    camera=dict(
                        eye=dict(x=eye_x, y=eye_y, z=z_height)
                    ),
                    xaxis=dict(range=[-1.5, 1.5]), # Keep scales consistent
                    yaxis=dict(range=[-1.5, 1.5]),
                    zaxis=dict(range=[-1.5, 1.5])
                )

                # Set a high resolution for the video
                fig.update_layout(width=1200, height=600, showlegend=False)

                png_path = temp_dir / f"frame_{step:08d}.png"
                fig.write_image(str(png_path))
                processed_frames.append(png_path)

                os.remove(downloaded_file.name)

            except Exception as e:
                print(f"Error at step {step}: {e}")

    if not processed_frames:
        return

    processed_frames.sort()

    # Stitching
    first_frame = cv2.imread(str(processed_frames[0]))
    h, w, _ = first_frame.shape
    video = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for frame_path in tqdm(processed_frames, desc="Compiling MP4"):
        video.write(cv2.imread(str(frame_path)))

    video.release()
    shutil.rmtree(temp_dir)
    print(f"Video saved as {output_name}")

if __name__ == "__main__":
    ENTITY = "tim-walter-tum"
    PROJECT = "Capability Maps"
    RUN_ID = "j9o8p1qw" # The 8-character ID found in the URL

    create_sphere_orbital_video(ENTITY, PROJECT, RUN_ID, rotations=2)