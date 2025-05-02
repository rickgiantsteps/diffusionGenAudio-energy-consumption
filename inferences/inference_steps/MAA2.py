# Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich
# Adapted from https://github.com/CompVis/latent-diffusion/blob/main/scripts/txt2img.py
#   LICENSE is in incl_licenses directory.
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Modifications”). 
# All Bytedance Inc.'s Modifications are Copyright (2023) Bytedance Inc..  
# *************************************************************************
import os
import torch
import glob
import numpy as np
from bigvgan.models import VocoderBigVGAN
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import argparse
import soundfile
from preprocess.n2s_by_openai import get_struct
from codecarbon import EmissionsTracker

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' # change to 'cpu‘ if you do not have gpu. generating with cpu is very slow.
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
        "--struct_prompt",
        type=str,
        default="",
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
    latent_width = int(duration * 31.5)
    if latent_width % 2 != 0:
        latent_width = (latent_width // 2 + 1) * 2
    return latent_width

def gen_wav(sampler,vocoder,prompt,struct_prompt,ddim_steps,scale,duration,n_samples):
    latent_width = dur_to_size(duration)
    start_code = torch.randn(n_samples, sampler.model.first_stage_model.embed_dim,latent_width).to(device=device, dtype=torch.float32)
    
    uc = None
    if scale != 1.0:
        emptycap = {'ori_caption':n_samples*[""],'struct_caption':n_samples*[""]}
        uc = sampler.model.get_learned_conditioning(emptycap)
    if struct_prompt == "":
        struct_prompt = get_struct(prompt)
    prompt_dict = {'ori_caption':n_samples *[prompt],'struct_caption':n_samples *[struct_prompt]}
    c = sampler.model.get_learned_conditioning(prompt_dict)
    shape = [sampler.model.first_stage_model.embed_dim, latent_width]  # 10 is latent height 
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
        wav = vocoder.vocode(spec.unsqueeze(0))
        if len(wav) < SAMPLE_RATE * duration:
            wav = np.pad(wav,SAMPLE_RATE*duration-len(wav),mode='constant',constant_values=0)
        wav_list.append(wav)
    return wav_list

if __name__ == '__main__':

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

    args = parse_args()

    sampler = initialize_model('configs/text2audio-ConcatDiT-ae1dnat_Skl20d2_struct2MLPanylen.yaml', 'useful_ckpts/maa2.ckpt')
    vocoder = VocoderBigVGAN('useful_ckpts/bigvgan',device=device)

    prompto = "Water running out of a faucet, some hitting a sink bottom, and some water filling a cup"
    struct = '<water running out of faucet & all>@<water hitting sink bottom & start>@<water filling a cup & mid>' 
    n_step = [10, 25, 50, 100, 150, 200]
    runs = 5

for k in range(runs):
    for x in n_step:

        tracker = EmissionsTracker(project_name=f"Make-an-Audio-2-inference_{x}-steps", tracking_mode="process", gpu_ids="1",
                                   output_dir=f'{parent_dir}/results/inference_steps/Make-an-Audio-2',
                                   output_file=f"Make-an-Audio-2-emissions-run{k + 1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with {x} steps")

        try:

            print("Generating audios, it may take a long time depending on your gpu performance")
            scl = 3; length = 10; samples = 1; save_name = "Make-an-Audio-2"
            wav_list = gen_wav(sampler,vocoder,prompt=prompto,struct_prompt=struct,ddim_steps=x,scale=scl,duration=length,n_samples=samples)

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        for idx,wav in enumerate(wav_list):
            soundfile.write(f'{current_dir}/genaudios/{save_name}_{x}-steps.wav',wav,samplerate=SAMPLE_RATE)
            print(f"audios are saved in {current_dir}/genaudios/Make-an-Audio-2/{save_name}_{x}-steps.wav")

emissions_base = glob.glob(f'{parent_dir}/results/inference_steps/Make-an-Audio-2/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
