<h1 align="center">ENN: Expressive Neural Network</h1>

<h2 align="center">A Neural Network Model with DCT Adaptive Activation Functions</h1>

<p align="center">
  <strong>
    <a href="https://scholar.google.com/citations?user=s3A8tT4AAAAJ&hl=en" target="_blank">Marc Martinez-Gost</a> · 
    <a href="https://scholar.google.es/citations?user=HPT8giEAAAAJ&hl=en" target="_blank">Ana Pérez Neira</a> · 
    <a href="https://www.cttc.cat/people/miguel-angel-lagunas/" target="_blank">Miguel Ángel Lagunas</a>
  </strong>
</p>

<div align="center">
  <a href="https://arxiv.org/abs/2307.00673"><img src="https://img.shields.io/badge/arXiv-ENN-%23b31b1b?style=flat&logo=arXiv" alt="arXiv"></a>
  &nbsp;
  <a href="https://ieeexplore.ieee.org/document/10418453"><img src="https://img.shields.io/badge/Xplore-ENN-%2300629B?style=flat&logo=IEEE" alt="IEEE Xplore"></a>
  &nbsp;
  <a href="https://drive.google.com/file/d/17ZshDBAn7PMp_GhE1iuvlAvqyhAuLjMz/view?usp=sharing"><img src="https://img.shields.io/badge/Colab-ENN-%23F9AB00?style=flat&logo=GoogleColab" alt="Google Colab"></a>
</div>
<br>

<p align="center">
<img width="914" alt="ENN" src="https://github.com/user-attachments/assets/41cc8130-ca4a-4c27-8dee-cb92834c02e2">
</p>



> The expressiveness of neural networks highly depends on the nature of the activation function, although these are usually assumed predefined and fixed during the training stage. Under a signal processing perspective, in this paper we present Expressive Neural Network (ENN), a novel model in which the non-linear activation functions are modeled using the Discrete Cosine Transform (DCT) and adapted using backpropagation during training. This parametrization keeps the number of trainable parameters low, is appropriate for gradient-based schemes, and adapts to different learning tasks. This is the first non-linear model for activation functions that relies on a signal processing perspective, providing high flexibility and expressiveness to the network. We contribute with insights in the explainability of the network at convergence by recovering the concept of bump, this is, the response of each activation function in the output space. Finally, through exhaustive experiments we show that the model can adapt to classification and regression tasks. The performance of ENN outperforms state of the art benchmarks, providing above a 40% gap in accuracy in some scenarios.

## Description
This repository contains a Python implementation of the ENN, which is a multilayer perceptron, whose activation functions are modeled with the DCT and learned during training. We offer the following options:
- [helloenn](https://github.com/marcmartinezgost/enn/edit/main/helloenn.ipynb): A Pytorch tutorial on the ENN and the DCT.
- You can also [run helloenn in Colab](https://colab.research.google.com/drive/1S70GaGfkSLipH_byNqAPnknESNzp5h_y?usp=drive_link).

We also provide an ENN model which is trained with the least mean squares (LMS) algorithm during backpropagation. This ensures a better learning. You can find implementation in [this notebook](https://github.com/marcmartinezgost/enn/edit/main/ENNwithLMS.ipynb) or
[run it in Colab](https://colab.research.google.com/drive/1e6Gtt2f3RU0a6bGukLi1XPDV8xw-lxDs?usp=drive_link).

## Installation
TBD

## Usage
TBD

## Citation
If you use this code in your work, please cite our [paper](https://ieeexplore.ieee.org/document/10418453):
``` bib
@article{martinez24enn,
    author={Martinez-Gost, Marc and Pérez-Neira, Ana and Lagunas, Miguel Ángel},
    journal={IEEE Journal of Selected Topics in Signal Processing}, 
    title={ENN: A Neural Network With DCT Adaptive Activation Functions}, 
    year={2024},
    volume={18},
    number={2},
    pages={232-241},
    doi={10.1109/JSTSP.2024.3361154}
  }
```
## Contact
If you have any questions, please contact marc.martinez@cttc.es or marc.martinez.gost@upc.edu.

## Acknowledgements
We thank Lucas Ventura in building and formatting this repository.

