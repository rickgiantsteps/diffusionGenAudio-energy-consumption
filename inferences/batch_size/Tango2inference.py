from codecarbon import EmissionsTracker
import soundfile as sf
from tango import Tango
import torch
import os
import glob

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tango = Tango("declare-lab/tango2-full")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tango.model = tango.model.to(device)  # Move the model to GPU

prompt = "Water running out of a faucet, some hitting a sink bottom, and some water filling a cup"
runs = 5

for k in range(runs):

    tracker = EmissionsTracker(project_name=f"Tango2-inference_batch10", tracking_mode="process", gpu_ids="1",
                                output_dir=f"{parent_dir}/results/batch_size/Tango2",
                                output_file=f"Tango2-emissions-batch10-run{k+1}.csv", allow_multiple_runs=True)
    tracker.start_task(f"Inference emissions with 10 waveforms per prompt")

    try:
        audio = tango.generate_for_batch(list(map(lambda _: prompt, range(10))), steps=100)
        model_emissions = tracker.stop_task()
    finally:
        _ = tracker.stop()

    for x in range(10):
        sf.write(f"{current_dir}/genaudios/Tango2/Tango2-batch10-n{x}.wav",
                audio[x], samplerate=16000)
    print(model_emissions)

for k in range(runs):
    file_counter = 0
    for i in range(2):
        tracker = EmissionsTracker(project_name=f"Tango2-inference_batch5", tracking_mode="process", gpu_ids="1",
                                   output_dir=f"{parent_dir}/results/batch_size/Tango2/batch5",
                                   output_file=f"Tango2-emissions-batch5-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 5 waveforms per prompt")

        try:
            audio = tango.generate_for_batch(list(map(lambda _: prompt, range(5))), steps=100)
            model_emissions = tracker.stop_task()
        finally:
            _ = tracker.stop()

        for x in range(5):
            sf.write(f"{current_dir}/genaudios/Tango2/Tango2-batch5-n{file_counter}.wav",
                    audio[x], samplerate=16000)
            file_counter += 1
    print(model_emissions)

for k in range(runs):
    file_counter = 0
    for i in range(5):
        tracker = EmissionsTracker(project_name=f"Tango2-inference_batch2", tracking_mode="process", gpu_ids="1",
                                   output_dir=f"{parent_dir}/results/batch_size/Tango2/batch2",
                                   output_file=f"Tango2-emissions-batch2-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 2 waveforms per prompt")

        try:
            audio = tango.generate_for_batch(list(map(lambda _: prompt, range(2))), steps=100)
            model_emissions = tracker.stop_task()
        finally:
            _ = tracker.stop()

        for x in range(2):
            sf.write(f"{current_dir}/genaudios/Tango2/Tango2-batch2-n{file_counter}.wav",
                    audio[x], samplerate=16000)
            file_counter += 1
    print(model_emissions)

for k in range(runs):
    for i in range(10):
        tracker = EmissionsTracker(project_name=f"Tango2-inference_batch1", tracking_mode="process", gpu_ids="1",
                                   output_dir=f"{parent_dir}/results/batch_size/Tango2/batch1",
                                   output_file=f"Tango2-emissions-batch1-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 1 waveforms per prompt")

        try:
            audio = tango.generate(prompt, steps=100)
            model_emissions = tracker.stop_task()
        finally:
            _ = tracker.stop()

        sf.write(f"{current_dir}/genaudios/Tango2/Tango2-batch1-n{i}.wav",
                audio, samplerate=16000)
    print(model_emissions)

emissions_base = glob.glob(f'{parent_dir}/results/batch_size/Tango2/emissions_base_*')
for f in emissions_base:
    os.remove(f)
emissions_base = glob.glob(f'{parent_dir}/results/batch_size/Tango2/batch5/emissions_base_*')
for f in emissions_base:
    os.remove(f)
emissions_base = glob.glob(f'{parent_dir}/results/batch_size/Tango2/batch2/emissions_base_*')
for f in emissions_base:
    os.remove(f)
emissions_base = glob.glob(f'{parent_dir}/results/batch_size/Tango2/batch1/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
