<div align="left">

# Inference scripts

</div>

This folder contains all the scripts used to collect energy consumption data of the 7 models for each of the three performed experiments.<br>

For the quality metrics experiment, the scripts necessary to compute the CLAP scores and FAD have also been included.<br>
The `getCLAP.py` file, contained in the CLAP folder, uses the 300 randomly chosen captions for each baseline to obtain the prompt coherence of the generations (using the Stable Audio Metrics library: https://github.com/Stability-AI/stable-audio-metrics). The FAD folder contains all the scripts necessary to get the objective quality measurement for all configurations (using the FADTK library: https://github.com/microsoft/fadtk).

## Install & Usage

The scripts related to AudioLDM, AudioLDM2, and Stable Audio Open can be run in the environment provided with this repository, as they are hosted on HuggingFace (https://huggingface.co/cvssp/audioldm-s-full-v2, https://huggingface.co/cvssp/audioldm2, https://huggingface.co/stabilityai/stable-audio-open-1.0). 
Stable Audio Open requires a HuggingFace user token for access. Ensure you are logged in via huggingface-cli login or have your token configured as an environment variable (HF_TOKEN) before running the model. Acceptance of the terms on the model's HF page is necessary to perform inferences.

For the remaining models, cloning their respective GitHub repositories, along with the creation of custom environments, is required: 

- **Make-an-Audio**: https://github.com/Text-to-Audio/Make-An-Audio
- **Make-an-Audio-2**: https://github.com/bytedance/Make-An-Audio-2
- **Tango/Tango2**: https://github.com/declare-lab/tango