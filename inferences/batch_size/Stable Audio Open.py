from diffusers import StableAudioPipeline
from codecarbon import EmissionsTracker
import soundfile as sf
import torch
import os
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Load the pre-trained pipeline
pipe = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float16,
    token="" # insert SAO token from HF
)
pipe = pipe.to(device="cuda", dtype=torch.float16)

# Define the prompt and other parameters
prompt = "Water running out of a faucet, some hitting a sink bottom, and some water filling a cup"
negative_prompt = "Low quality."
generator = torch.Generator("cuda").manual_seed(0)

# Number of runs
runs = 5

for k in range(runs):

    tracker = EmissionsTracker(project_name=f"Stable Audio Open-inference_batch10", tracking_mode="process", gpu_ids="1",
                               output_dir=f"{parent_dir}/results/batch_size/Stable Audio Open",
                               output_file=f"Stable Audio Open-emissions-batch10-run{k+1}.csv", allow_multiple_runs=True)
    tracker.start_task(f"Inference emissions with 10 waveforms per prompt")

    try:

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=100,
            audio_end_in_s=10.0,
            num_waveforms_per_prompt=10,  # Batch size
            generator=generator,
        )
        audio_batch = result.audios

        model_emissions = tracker.stop_task()

    finally:
        _ = tracker.stop()

    for x in range(10):
        output = audio_batch[x].T.float().cpu().numpy()  # Convert Torch tensor to NumPy array
        output_file = f"{current_dir}/genaudios/Stable Audio Open/Stable Audio Open-batch10-n{x}.wav"
        sf.write(output_file, output, pipe.vae.sampling_rate)
        print(f"Saved: {output_file}")
        print(f"Emissions data: {model_emissions}")

for k in range(runs):
    file_counter = 0
    for i in range(2):
        tracker = EmissionsTracker(project_name=f"Stable Audio Open-inference_batch5", tracking_mode="process", gpu_ids="1",
                                   output_dir=f"{parent_dir}/results/batch_size/Stable Audio Open/batch5",
                                   output_file=f"Stable Audio Open-emissions-batch5-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 5 waveforms per prompt")

        try:

            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=100,
                audio_end_in_s=10.0,
                num_waveforms_per_prompt=5,  # Batch size
                generator=generator,
            )
            audio_batch = result.audios

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        for x in range(5):
            output = audio_batch[x].T.float().cpu().numpy()  # Convert Torch tensor to NumPy array
            output_file = f"{current_dir}/genaudios/Stable Audio Open/Stable Audio Open-batch5-n{file_counter}.wav"
            sf.write(output_file, output, pipe.vae.sampling_rate)
            file_counter += 1
            print(f"Saved: {output_file}")
            print(f"Emissions data: {model_emissions}")

# Batch size for audio generation
batch_size = 2

for k in range(runs):
    file_counter = 0
    for i in range(5):
        tracker = EmissionsTracker(
            project_name=f"Stable Audio Open-inference_batch{batch_size}",
            tracking_mode="process", gpu_ids="1",
            output_dir=f"{parent_dir}/results/batch_size/Stable Audio Open/batch{batch_size}",
            output_file=f"Stable Audio Open-emissions-batch{batch_size}-n{i}-run{k+1}.csv",
            allow_multiple_runs=True
        )
        tracker.start_task(f"Inference emissions with {batch_size} waveforms per prompt")

        try:
            # Generate multiple audio samples in one inference step
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=100,
                audio_end_in_s=10.0,
                num_waveforms_per_prompt=batch_size,  # Batch size
                generator=generator,
            )
            audio_batch = result.audios

            # Stop tracker and record emissions
            model_emissions = tracker.stop_task()

        finally:
            tracker.stop()

        # Save the generated audio samples
        for x in range(batch_size):
            output = audio_batch[x].T.float().cpu().numpy()  # Convert Torch tensor to NumPy array
            output_file = f"{current_dir}/genaudios/Stable Audio Open/Stable Audio Open-batch{batch_size}-n{file_counter}.wav"
            sf.write(output_file, output, pipe.vae.sampling_rate)
            file_counter += 1
            print(f"Saved: {output_file}")
            print(f"Emissions data: {model_emissions}")


batch_size = 1

for k in range(runs):
    file_counter = 0
    for i in range(10):
        tracker = EmissionsTracker(
            project_name=f"Stable Audio Open-inference_batch{batch_size}",
            tracking_mode="process", gpu_ids="1",
            output_dir=f"{parent_dir}/results/batch_size/Stable Audio Open/batch{batch_size}",
            output_file=f"Stable Audio Open-emissions-batch{batch_size}-n{i}-run{k+1}.csv",
            allow_multiple_runs=True
        )
        tracker.start_task(f"Inference emissions with {batch_size} waveform per prompt")

        try:
            # Generate multiple audio samples in one inference step
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=100,
                audio_end_in_s=10.0,
                num_waveforms_per_prompt=batch_size,  # Batch size
                generator=generator,
            )
            audio_batch = result.audios

            # Stop tracker and record emissions
            model_emissions = tracker.stop_task()

        finally:
            tracker.stop()

        # Save the generated audio samples
        for x in range(batch_size):
            output = audio_batch[x].T.float().cpu().numpy()  # Convert Torch tensor to NumPy array
            output_file = f"{current_dir}/genaudios/Stable Audio Open/Stable Audio Open-batch{batch_size}-n{i}.wav"
            sf.write(output_file, output, pipe.vae.sampling_rate)
            print(f"Saved: {output_file}")
            print(f"Emissions data: {model_emissions}")

# Cleanup base emissions files
emissions_base = glob.glob(f'{parent_dir}/results/batch_size/Stable Audio Open/emissions_base_*')
for f in emissions_base:
    os.remove(f)
emissions_base = glob.glob(f'{parent_dir}/results/batch_size/Stable Audio Open/batch5/emissions_base_*')
for f in emissions_base:
    os.remove(f)
emissions_base = glob.glob(f'{parent_dir}/results/batch_size/Stable Audio Open/batch2/emissions_base_*')
for f in emissions_base:
    os.remove(f)
emissions_base = glob.glob(f'{parent_dir}/results/batch_size/Stable Audio Open/batch1/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
