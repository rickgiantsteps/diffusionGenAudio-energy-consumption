import torch
from codecarbon import EmissionsTracker
from itertools import chain
import numpy as np
from vocoder.bigvgan.models import VocoderBigVGAN
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import argparse
import soundfile
from tqdm import tqdm
import pandas as pd
import os
import glob

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = 'cuda'
SAMPLE_RATE = 16000

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="a bird chirps",
        help="the prompt to generate audio"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="audio duration, seconds",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=3.0, # if it's 1, only condition is taken into consideration
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    ) 

    parser.add_argument(
        "--save_name",
        type=str,
        default='test', 
        help="audio path name for saving",
    ) 
    return parser.parse_args()

def initialize_model(config, ckpt,device=device):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt,map_location='cpu')["state_dict"], strict=False)

    model = model.to(device)
    model.cond_stage_model.to(model.device)
    model.cond_stage_model.device = model.device
    print(model.device,device,model.cond_stage_model.device)
    sampler = DDIMSampler(model)

    return sampler

def dur_to_size(duration):
    latent_width = int(duration * 7.8)
    if latent_width % 4 != 0:
        latent_width = (latent_width // 4 + 1) * 4
    return latent_width

def gen_wav(sampler,vocoder,prompt,ddim_steps,scale,duration,n_samples):
    latent_width = dur_to_size(duration)
    start_code = torch.randn(n_samples, sampler.model.first_stage_model.embed_dim, 10, latent_width).to(device=device, dtype=torch.float32)
    
    uc = None
    if scale != 1.0:
        uc = sampler.model.get_learned_conditioning(n_samples * [""])
    c = sampler.model.get_learned_conditioning(n_samples * [prompt])
    shape = [sampler.model.first_stage_model.embed_dim, 10, latent_width]  # 10 is latent height 
    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                        conditioning=c,
                                        batch_size=n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc,
                                        x_T=start_code)

    x_samples_ddim = sampler.model.decode_first_stage(samples_ddim)

    wav_list = []
    for idx,spec in enumerate(x_samples_ddim):
        wav = vocoder.vocode(spec)
        if len(wav) < SAMPLE_RATE * duration:
            wav = np.pad(wav,SAMPLE_RATE*duration-len(wav),mode='constant',constant_values=0)
        wav_list.append(wav)
    return wav_list

if __name__ == '__main__':

    args = parse_args()

    sampler = initialize_model('configs/text_to_audio/txt2audio_args.yaml', 'useful_ckpts/maa1_full.ckpt')
    vocoder = VocoderBigVGAN('useful_ckpts/bigvgan',device=device)

    n_step = [10, 25, 50, 100, 150, 200]
    datasets = ["audiocaps_captions.csv", "clotho_captions.csv"]

    for k in datasets:

        df = pd.read_csv(f"{current_dir}/CLAP/{k}")
        filenames = df["file_name"].tolist()
        captions = df["caption"].tolist()
        durations = df["duration"].tolist() 

        for i, file in enumerate(tqdm(filenames, desc=f"Processing MAA dataset {k}", unit="item")):
            for x in n_step:

                if os.path.exists(f'{current_dir}/genaudios/MAA/{k.split("_")[0]}/{x}/MAA_{x}-steps-'+file):               
                    print(f"{file} already exists, skipping.")
                    continue
                else:

                    tracker = EmissionsTracker(project_name=f"MAA_{k.split('_')[0]}-{x}-steps-{file.split('.')[0]}", 
                        tracking_mode="process", gpu_ids = "2",
                        output_dir=f"{parent_dir}/results/quality_metrics/MAA",
                        output_file=f"MAA_{k.split('_')[0]}.csv", allow_multiple_runs=True)
                    tracker.start_task(f"Inference emissions with {x} steps, for {file}")

                    wav_list = gen_wav(sampler,vocoder,prompt=captions[i],ddim_steps=x,scale=3,duration=10,n_samples=1)

                    model_emissions = tracker.stop_task()
                    _ = tracker.stop()

                    for idx,wav in enumerate(wav_list):
                        soundfile.write(f"{current_dir}/genaudios/MAA/{k.split('_')[0]}/{x}/MAA_{x}-steps-{file}",wav,samplerate=SAMPLE_RATE)
                        print(f"Sample with {x} steps with prompt: '{captions[i]}'")

        print("Done!")

emissions_base = glob.glob(f'{parent_dir}/results/quality_metrics/MAA/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
