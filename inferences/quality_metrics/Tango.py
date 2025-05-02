import torch
import os
from tqdm import tqdm
import pandas as pd
from codecarbon import EmissionsTracker
import glob
import soundfile as sf
from tango import Tango

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
datasets = ["clotho_captions.csv", "audiocaps_captions.csv"]

tango = Tango("declare-lab/tango-full")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tango.model = tango.model.to(device)  # Move the model to GPU

for k in datasets:

    df = pd.read_csv(f"{current_dir}/CLAP/{k}")
    filenames = df["file_name"].tolist()
    captions = df["caption"].tolist()
    durations = df["duration"].tolist() 

    n_steps = [10, 25, 50, 100, 150, 200]

    for i, file in enumerate(tqdm(filenames, desc=f"Processing Tango dataset {k}", unit="item")):
        for x in n_steps:
            if os.path.exists(f'{current_dir}/genaudios/Tango/{k.split("_")[0]}/{x}/Tango_{x}-steps-'+file):
                print(f"{file} already exists, skipping.")
                continue
            else:

                tracker = EmissionsTracker(project_name=f"Tango_{k.split('_')[0]}-{x}-steps-{file.split('.')[0]}", 
                    tracking_mode="process", gpu_ids = "2",
                    output_dir=f"{parent_dir}/results/quality_metrics/Tango",
                    output_file=f"Tango_{k.split('_')[0]}.csv", allow_multiple_runs=True)
                tracker.start_task(f"Inference emissions with {x} steps, for {file}")

                audio = tango.generate(captions[i], steps=x)

                model_emissions = tracker.stop_task()
                _ = tracker.stop()
                
                sf.write(f"{current_dir}/genaudios/Tango/{k.split('_')[0]}/{x}/Tango_{x}-steps-{file}", audio, samplerate=16000)
                print(f"Sample with {x} steps with prompt: '{captions[i]}'")

    print("Done!")

emissions_base = glob.glob(f'{parent_dir}/results/quality_metrics/Tango/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
