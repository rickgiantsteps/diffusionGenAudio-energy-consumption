<div align="center">

# Diffused Responsibility: A Comprehensive Energy Consumption Analysis of Generative Audio Diffusion Models
 
[Author]()<sup>1</sup>

<sup>1</sup> Affiliation <br>

</div>

- [Abstract](#abstract)
- [Install & Usage](#install--usage)
- [Additional information](#additional-information)
    
## Abstract

Text-to-audio models have recently emerged as a powerful technology for generating sound from textual descriptions. However, their high computational demands raise concerns about energy consumption and environmental impact. In this paper, we conduct a comprehensive analysis of the energy usage of 7 state-of-the-art text-to-audio diffusion-based generative models, evaluating to what extent variations in generation parameters affect energy consumption at inference time. We also aim to identify an optimal balance between audio quality and energy consumption by considering Pareto-optimal solutions across all selected models. Our findings provide insights into the trade-offs between performance and environmental impact, contributing to the development of more efficient generative audio models.

## Install & Usage

In order to run the Jupyter notebooks, you need to clone the repo, create a virtual environment, and install the needed packages.

You can create the virtual environment and install the needed packages using conda with the following command: 

```
conda env create -f requirements.yml
```

Once everything is installed, you can run the Jupyter Notebook following the instruction reported on it and reproduce the results. 

<br>
The scripts contained in the 'inferences' folder can be run by creating environments specific to the desired model; further information is provided in the folder's README.


## Additional information

For more details:
"[Diffused Responsibility: A Comprehensive Energy Consumption Analysis of Generative Audio Diffusion Models]()", Author

If you use code or comments from this work, please cite:

```BibTex
@inproceedings{author2025consumption,
  title={Diffused Responsibility: A Comprehensive Energy Consumption Analysis of 
  Generative Audio Diffusion Models},
  author={Author},
  year={2025}
}
```

