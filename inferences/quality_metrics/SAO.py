from codecarbon import EmissionsTracker
from diffusers import StableAudioPipeline
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import torch
import glob
import os

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
datasets = ["audiocaps_captions.csv", "clotho_captions.csv"]

pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16,
                                        token="") # insert SAO token from HF
pipe = pipe.to(device="cuda", dtype=torch.float16)

negative_prompt = "Low quality."
generator = torch.Generator("cuda").manual_seed(0)

for k in datasets:

    df = pd.read_csv(f"{current_dir}/CLAP/{k}")
    filenames = df["file_name"].tolist()
    captions = df["caption"].tolist()
    durations = df["duration"].tolist()

    n_step = [10, 25, 50, 100, 150, 200]

    for i, file in enumerate(tqdm(filenames, desc=f"Processing SAO dataset {k}", unit="item")):
        for x in n_step:
                if os.path.exists(f'{current_dir}/genaudios/SAO/{k.split("_")[0]}/{x}/SAO_{x}-steps-'+file):               
                        print(f"{file} already exists, skipping.")
                        continue
                else:

                        tracker = EmissionsTracker(project_name=f"SAO_{k.split('_')[0]}-{x}-steps-{file.split('.')[0]}", 
                            tracking_mode="process", gpu_ids = "1",
                            output_dir=f"{parent_dir}/results/quality_metrics/SAO",
                            output_file=f"SAO_{k.split('_')[0]}.csv", allow_multiple_runs=True)
                        tracker.start_task(f"Inference emissions with {x} steps, for {file}")

                        # run the generation
                        audio = pipe(
                                captions[i],
                                negative_prompt=negative_prompt,
                                num_inference_steps=x,
                                audio_end_in_s=10,
                                num_waveforms_per_prompt=1,
                                generator=generator,
                        ).audios

                        model_emissions = tracker.stop_task()
                        _ = tracker.stop()

                        output = audio[0].T.float().cpu().numpy()
                        sf.write(
                        f"{current_dir}/genaudios/SAO/{k.split('_')[0]}/{x}/SAO_{x}-steps-{file}",
                                output, pipe.vae.sampling_rate)

emissions_base = glob.glob(f'{parent_dir}/results/quality_metrics/SAO/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
