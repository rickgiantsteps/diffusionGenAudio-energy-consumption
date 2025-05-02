from codecarbon import EmissionsTracker
from diffusers import AudioLDM2Pipeline
from tqdm import tqdm
import pandas as pd
import scipy
import torch
import glob
import os

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
datasets = ["audiocaps_captions.csv", "clotho_captions.csv"]

repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to(device="cuda", dtype=torch.float16)
negative_prompt = "Low quality."
generator = torch.Generator("cuda").manual_seed(0)

for k in datasets:

    df = pd.read_csv(f"{current_dir}/CLAP/{k}")
    filenames = df["file_name"].tolist()
    captions = df["caption"].tolist()
    durations = df["duration"].tolist()

    n_step = [10, 25, 50, 100, 150, 200]

    for i, file in enumerate(tqdm(filenames, desc=f"Processing AudioLDM2 dataset {k}", unit="item")):
        for x in n_step:
            if os.path.exists(f'{current_dir}/genaudios/AudioLDM2/{k.split("_")[0]}/{x}/AudioLDM2_{x}-steps-'+file):                       
                print(f"{file} already exists, skipping.")
                continue
            else:

                tracker = EmissionsTracker(project_name=f"AudioLDM2_{k.split('_')[0]}-{x}-steps-{file.split('.')[0]}",
                    tracking_mode="process", gpu_ids = "1",
                    output_dir=f"{parent_dir}/results/quality_metrics/AudioLDM2",
                    output_file=f"AudioLDM2_{k.split('_')[0]}.csv", allow_multiple_runs=True)
                tracker.start_task(f"Inference emissions with {x} steps, for {file}")

                # run the generation
                audio = pipe(
                        captions[i],
                        negative_prompt=negative_prompt,
                        num_inference_steps=x,
                        audio_length_in_s=10,
                        num_waveforms_per_prompt=1,
                        generator=generator,
                ).audios

                model_emissions = tracker.stop_task()
                _ = tracker.stop()

                scipy.io.wavfile.write(
                    f"{current_dir}/genaudios/AudioLDM2/{k.split('_')[0]}/{x}/AudioLDM2_{x}-steps-{file}",
                    rate=16000, data=audio[0])
                print(f"{x} steps, {captions[i]}")

emissions_base = glob.glob(f'{parent_dir}/results/quality_metrics/AudioLDM2/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
