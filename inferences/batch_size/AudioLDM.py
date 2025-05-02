from diffusers import AudioLDMPipeline
from codecarbon import EmissionsTracker
import scipy
import torch
import os
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

repo_id = "cvssp/audioldm"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to(device="cuda", dtype=torch.float16)

prompt = "Water running out of a faucet, some hitting a sink bottom, and some water filling a cup"
negative_prompt = "Low quality."
generator = torch.Generator("cuda").manual_seed(0)

runs = 5

for k in range(runs):

    tracker = EmissionsTracker(project_name=f"AudioLDM-inference_batch10", tracking_mode="process", gpu_ids="1",
                               output_dir=f'{parent_dir}/results/batch_size/AudioLDM',
                               output_file=f"AudioLDM-emissions-batch10-run{k+1}.csv", allow_multiple_runs=True)
    tracker.start_task(f"Inference emissions with 10 waveforms per prompt")

    try:

        audio = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=100,
            audio_length_in_s=10.0,
            num_waveforms_per_prompt=10,
            generator=generator,
        ).audios

        model_emissions = tracker.stop_task()

    finally:
        _ = tracker.stop()

    for i in range(10):
        scipy.io.wavfile.write(
            f"{current_dir}/genaudios/AudioLDM/AudioLDM-batch10-n{i}.wav",
            rate=16000, data=audio[i])
        print(model_emissions)

for k in range(runs):
    file_counter = 0
    for i in range(2):
        tracker = EmissionsTracker(project_name=f"AudioLDM-inference_batch5", tracking_mode="process", gpu_ids="1",
                                   output_dir=f"{parent_dir}/results/batch_size/AudioLDM/batch5",
                                   output_file=f"AudioLDM-emissions-batch5-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 5 waveforms per prompt")

        try:

            audio = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=100,
                audio_length_in_s=10.0,
                num_waveforms_per_prompt=5,
                generator=generator,
            ).audios

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        for x in range(5):
            scipy.io.wavfile.write(
                f"{current_dir}/genaudios/AudioLDM/AudioLDM-batch5-n{file_counter}.wav",
                rate=16000, data=audio[x])
            file_counter += 1
        print(model_emissions)


for k in range(runs):
    file_counter = 0
    for i in range(5):
        tracker = EmissionsTracker(project_name=f"AudioLDM-inference_batch2", tracking_mode="process", gpu_ids="1",
                                   output_dir=f"{parent_dir}/results/batch_size/AudioLDM/batch2",
                                   output_file=f"AudioLDM-emissions-batch2-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 2 waveforms per prompt")

        try:

            audio = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=100,
                audio_length_in_s=10.0,
                generator=generator,
                num_waveforms_per_prompt=2,
            ).audios

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        for x in range(2):
            scipy.io.wavfile.write(
            f"{current_dir}/genaudios/AudioLDM/AudioLDM-batch2-n{file_counter}.wav",
            rate=16000, data=audio[x])
            file_counter += 1
        print(model_emissions)

for k in range(runs):
    for i in range(10):
        tracker = EmissionsTracker(project_name=f"AudioLDM-inference_batch1", tracking_mode="process", gpu_ids="1",
                                   output_dir=f"{parent_dir}/results/batch_size/AudioLDM/batch1",
                                   output_file=f"AudioLDM-emissions-batch1-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 1 waveform per prompt")

        try:

            audio = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=100,
                audio_length_in_s=10.0,
                num_waveforms_per_prompt=1,
                generator=generator,
            ).audios

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        scipy.io.wavfile.write(
            f"{current_dir}/genaudios/AudioLDM/AudioLDM-batch1-n{i}.wav",
            rate=16000, data=audio[0])
        print(model_emissions)


emissions_base = glob.glob(f'{parent_dir}/results/batch_size/AudioLDM/emissions_base_*')
for f in emissions_base:
    os.remove(f)
emissions_base = glob.glob(f'{parent_dir}/results/batch_size/AudioLDM/batch5/emissions_base_*')
for f in emissions_base:
    os.remove(f)
emissions_base = glob.glob(f'{parent_dir}/results/batch_size/AudioLDM/batch2/emissions_base_*')
for f in emissions_base:
    os.remove(f)
emissions_base = glob.glob(f'{parent_dir}/results/batch_size/AudioLDM/batch1/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
