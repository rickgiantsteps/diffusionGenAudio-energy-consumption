from diffusers import AudioLDMPipeline
from codecarbon import EmissionsTracker
import scipy
import torch
import os
import glob

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to(device="cuda", dtype=torch.float16)

prompt = "Water running out of a faucet, some hitting a sink bottom, and some water filling a cup"
negative_prompt = "Low quality."
generator = torch.Generator("cuda").manual_seed(0)

n_step = [10, 25, 50, 100, 150, 200]
runs = 5

for k in range(runs):
    for x in n_step:

        tracker = EmissionsTracker(project_name=f"AudioLDM-inference_{x}-steps", tracking_mode="process", gpu_ids="1",
                                   output_dir=f'{parent_dir}/results/inference_steps/AudioLDM',
                                   output_file=f"AudioLDM-emissions-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with {x} steps")

        try:
            # run the generation
            audio = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=x,
                audio_length_in_s=10.0,
                num_waveforms_per_prompt=1,
                generator=generator,
            ).audios

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        scipy.io.wavfile.write(
            f"{current_dir}/genaudios/AudioLDM/AudioLDM_{x}-steps.wav",
            rate=16000, data=audio[0])
        print(model_emissions)

emissions_base = glob.glob(f'{parent_dir}/results/inference_steps/AudioLDM/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")