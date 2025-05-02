import torch
import os
import glob
from codecarbon import EmissionsTracker
import soundfile as sf
from tango import Tango

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

tango = Tango("declare-lab/tango2-full")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tango.model = tango.model.to(device)  # Move the model to GPU

prompt = "Water running out of a faucet, some hitting a sink bottom, and some water filling a cup"

n_step = [10, 25, 50, 100, 150, 200]
runs = 5

for k in range(runs):
    for x in n_step:

        tracker = EmissionsTracker(project_name=f"Tango2-inference_{x}-steps", tracking_mode="process", gpu_ids="1",
                                   output_dir=f"{parent_dir}/results/inference_steps/Tango2",
                                   output_file=f"Tango2-emissions-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with {x} steps")

        try:

            audio = tango.generate(prompt, steps=x)

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        sf.write(f"{current_dir}/genaudios/Tango2/Tango2_{x}-steps.wav",
                 audio, samplerate=16000)

        print(model_emissions)

emissions_base = glob.glob(f'{parent_dir}/results/inference_steps/Tango2/emissions_base_*')
for f in emissions_base:
    os.remove(f)
