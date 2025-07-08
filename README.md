<div align="center">

# Diffused Responsibility: Analyzing the Energy Consumption of Generative Text-to-Audio Diffusion Models
 
[Riccardo Passoni](https://www.linkedin.com/in/riccardo-passoni/?locale=en_US)<sup>1</sup>, [Francesca Ronchini](https://www.linkedin.com/in/francesca-ronchini/)<sup>1</sup>, [Luca Comanducci](https://lucacoma.github.io/)<sup>1</sup>, [Romain Serizel](https://members.loria.fr/RSerizel/)<sup>2</sup>, [Fabio Antonacci](https://www.deib.polimi.it/ita/personale/dettagli/573870)<sup>1</sup>

<sup>1</sup> Dipartimento di Elettronica, Informazione e Bioingegneria - Politecnico di Milano, Milan, Italy <br>
<sup>2</sup> Université de Lorraine, CNRS, Inria, Loria, Nancy, France <br>

</div>

- [Abstract](#abstract)
- [Install & Usage](#install--usage)
- [Additional information](#additional-information)
    
## Abstract

Text-to-audio models have recently emerged as a powerful technology for generating sound from textual descriptions. However, their high computational demands raise concerns about energy consumption and environmental impact. In this paper, we conduct an analysis of the energy usage of 7 state-of-the-art text-to-audio diffusion-based generative models, evaluating to what extent variations in generation parameters affect energy consumption at inference time. We also aim to identify an optimal balance between audio quality and energy consumption by considering Pareto-optimal solutions across all selected models. Our findings provide insights into the trade-offs between performance and environmental impact, contributing to the development of more efficient generative audio models.

## Install & Usage

In order to run the Jupyter notebooks, you need to clone the repo, create a virtual environment, and install the needed packages.

You can create the virtual environment and install the needed packages using conda with the following command: 

```
conda env create -f requirements.yml
```

Once everything is installed, you can run the Jupyter Notebook following the instruction reported on it and reproduce the results. <br>

The scripts contained in the 'inferences' folder can be run by creating environments specific to the desired model; further information is provided in the folder's README.
The 'sanitycheck' folder contains a brief confirmation of the statistical significance of the quality metrics experiment.


## Additional information

For more details:
"[Diffused Responsibility: Analyzing the Energy Consumption of Generative Text-to-Audio Diffusion Models]()" (Riccardo Passoni, Francesca Ronchini, Luca Comanducci, Romain Serizel, Fabio Antonacci)

If you use code or comments from this work, please cite:

```BibTex
@misc{passoni2025diffusedresponsibility,
      title={Diffused Responsibility: Analyzing the Energy Consumption of Generative Text-to-Audio Diffusion Models}, 
      author={Riccardo Passoni and Francesca Ronchini and Luca Comanducci and Romain Serizel and Fabio Antonacci},
      year={2025},
      eprint={2505.07615},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2505.07615}, 
}
```

