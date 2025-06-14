<div align="left">

# Sanity check

</div>

This folder contains the results of a brief sanity check to see whether using 300‑sample subsets of the two datasets for the quality‑metrics experiment affects the results.<br>

The results for **AudioLDM** and **Stable Audio Open** are here available using the entirety of the *Clotho-eval* and *AudioCaps-test* datasets, as well as a brief statistical analysis of the correlation between these additional results and the ones contained in the paper.
To make the correlation between the results easier to understand at a glance, a plot containing both sets of experiments for the two models is also included.

Although absolute scores shift slightly, the patterns remain unchanged: Pearson correlations exceed 0.94 (p < 0.005) and ANCOVA tests return p > 0.92, indicating no meaningful difference in slope.
Together, these files demonstrate that a 300‑sample subset is a reliable stand‑in for full‑dataset evaluation.