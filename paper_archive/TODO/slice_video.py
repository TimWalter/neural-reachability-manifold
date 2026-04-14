import wandb
import cv2
import os
import shutil
import json
import plotly.io as pio
from pathlib import Path
from tqdm import tqdm

def create_run_timelapse(entity, project, run_id, output_name="slice_evolution.mp4", fps=5):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)

    print(f"Fetching history for run: {run.name}")
    # The key in your logger.py is "Boundaries/Slice"
    history = run.scan_history(keys=["Boundaries/Slice", "_step"])

    processed_frames = []

    for row in tqdm(history, desc="Processing Plotly frames"):
        if "Boundaries/Slice" in row:
            step = row["_step"]
            image_meta = row["Boundaries/Slice"]

            # 1. Download the artifact (likely a .plotly.json file)
            file_path = image_meta["path"]
            downloaded_file = run.file(file_path).download(root=temp_dir, replace=True)

            # 2. Convert Plotly JSON to a real PNG
            try:
                with open(downloaded_file.name, 'r') as f:
                    fig_dict = json.load(f)

                fig = pio.from_json(json.dumps(fig_dict))

                # Update layout for video (e.g., set fixed size)
                fig.update_layout(width=1000, height=1000)

                png_path = temp_dir / f"frame_{step:08d}.png"
                fig.write_image(str(png_path))
                processed_frames.append(png_path)

                # Cleanup the JSON file to save space
                os.remove(downloaded_file.name)

            except Exception as e:
                print(f"Failed to process step {step}: {e}")

    if not processed_frames:
        print("No valid frames were processed.")
        return

    processed_frames.sort()

    # 3. Use OpenCV to stitch the newly created PNGs
    first_frame = cv2.imread(str(processed_frames[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    for frame_path in tqdm(processed_frames, desc="Compiling video"):
        img = cv2.imread(str(frame_path))
        video.write(img)

    video.release()
    print(f"Successfully saved {output_name}")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Replace these with your actual W&B details
    ENTITY = "tim-walter-tum"
    PROJECT = "Capability Maps"
    RUN_ID = "j9o8p1qw" # The 8-character ID found in the URL

    create_run_timelapse(ENTITY, PROJECT, RUN_ID)