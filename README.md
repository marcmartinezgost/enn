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
  <a href="https://colab.research.google.com/drive/1S70GaGfkSLipH_byNqAPnknESNzp5h_y?usp=drive_link"><img src="https://img.shields.io/badge/Colab-ENN-%23F9AB00?style=flat&logo=GoogleColab" alt="Google Colab"></a>
</div>
<br>

<p align="center">
<img width="914" alt="ENN" src="https://github.com/user-attachments/assets/41cc8130-ca4a-4c27-8dee-cb92834c02e2">
</p>



> The expressiveness of neural networks highly depends on the nature of the activation function, although these are usually assumed predefined and fixed during the training stage. Under a signal processing perspective, in this paper we present Expressive Neural Network (ENN), a novel model in which the non-linear activation functions are modeled using the Discrete Cosine Transform (DCT) and adapted using backpropagation during training. This parametrization keeps the number of trainable parameters low, is appropriate for gradient-based schemes, and adapts to different learning tasks. This is the first non-linear model for activation functions that relies on a signal processing perspective, providing high flexibility and expressiveness to the network. We contribute with insights in the explainability of the network at convergence by recovering the concept of bump, this is, the response of each activation function in the output space. Finally, through exhaustive experiments we show that the model can adapt to classification and regression tasks. The performance of ENN outperforms state of the art benchmarks, providing above a 40% gap in accuracy in some scenarios.

## Description
This repository contains a Python implementation of the ENN, which is a multilayer perceptron, whose activation functions are modeled with the DCT and learned during training. We offer the following options:
- [helloenn](https://github.com/marcmartinezgost/enn/blob/main/helloenn.ipynb): A Pytorch tutorial on the ENN and the DCT.
- You can also [run helloenn in Colab](https://colab.research.google.com/drive/1S70GaGfkSLipH_byNqAPnknESNzp5h_y?usp=drive_link).

We also provide an ENN model which is trained with the least mean squares (LMS) algorithm during backpropagation. This ensures a better learning. You can find implementation in [this notebook](https://github.com/marcmartinezgost/enn/blob/main/ENNwithLMS.ipynb) or
[run it in Colab](https://colab.research.google.com/drive/1e6Gtt2f3RU0a6bGukLi1XPDV8xw-lxDs?usp=drive_link).

<!-- ## Installation section -->

## Pytorch model
The ENN is an MLP and can be easily implemented in Pytorch. We implement the ENN with a single hidden layer, although the number of layers can be increased arbitrarily. The class ```ENN()``` contains two functions. 

<details>
  <summary> __init__() function </summary>

  ``` python
      def __init__(self, input_dim, hidden_dim, output_dim, n_coeffs=6, nft=512):
          """
          This function creates the ENN with a single hidden layer. The adaptive activation functions initialized as lines.
      
          Parameters
          ----------
          input_dim : int
              Number of input variables.
          hidden_dim : int
              Number of neurons in the hidden layer.
          output_dim : int
              Number of output neurons in the output layer.
          n_coeffs: int
              Number of DCT coefficients per neuron. Same for all neurons.
          nft : int
              Resolution of the DCT.
      
          Returns
          -------
          ENN object
              ENN model.
          """
          super(ENN, self).__init__()
          # Linear weights
          self.fc1 = nn.Linear(input_dim, hidden_dim)
          self.fc2 = nn.Linear(hidden_dim, output_dim)
  
          # Adaptive Activation Functions
          idx = torch.arange(2, 2*n_coeffs+1, 2)
          coeffs = (2/nft) * torch.tensor([-207.6068,-23.1526,-8.3292,-4.2629,-2.5883,-1.5147])[:n_coeffs]
          F = torch.tile(coeffs,(hidden_dim,1))
          self.F_hid = nn.parameter.Parameter(data=F, requires_grad=True)
          self.F_out = nn.parameter.Parameter(data=coeffs[None,...], requires_grad=True)
          self.Q_hid = torch.tile(idx, (hidden_dim,1))
          self.Q_out = idx[None,...]
          self.nft = nft
  ```
  
  The ```ENN``` object contains two linear layers of weights. Each layer also contains a set of attributes for the activation functions. For instance, the hidden layer contains:
  - F_hid: DCT coefficients of AAF in the hidden layer. These are initialized as lines.
  - Q_hid: Indeces of the DCT coefficients in the hidden layer. These are fixed and equal for all neurons in the ENN.
  The same attributes exist for the output layer (namely, ```F_out``` and ```Q_out```). Note that ```F_hid``` and ```F_out``` are ```Parameter``` objects with ```requires_grad=True``` to ensure that the forward information is used in backpropagation.


</details>
<details>
  <summary> Forward function </summary>
 
  ``` python
      def forward(self, x):
          """
          This function creates the ENN with a single hidden layer. The adaptive activation functions initialized as lines.
      
          Parameters
          ----------
          x : Torch tensor
              Input batch of data.
          Returns
          -------
          out : Torch tensor
              Output of the ENN.
          """
          # Linear layer at the hidden layer
          out_fc1 = self.fc1(x)[...,None]
  
          # AAF at the hidden layer
          scaled = ((out_fc1 + 1) / 2) * self.nft
          dct_basis = self.F_hid * torch.cos( (np.pi / (2 * self.nft)) * (self.Q_hid-1) * (2*scaled-1) )
          hid_out = torch.sum(dct_basis, -1, keepdims=False)
  
          # Linear function at the output layer
          out_fc2 = self.fc2(hid_out)[...,None]
  
          # AAF at the output layer
          scaled_out = ((out_fc2 + 1) / 2) * self.nft
          dct_basis_out = self.F_out * torch.cos( (np.pi / (2 * self.nft)) * (self.Q_out-1) * (2*scaled_out-1) )
          out = torch.sum(dct_basis_out, -1, keepdims=False)
  
          return out
  ```
  
  The activation function at layer $l\in\{0,1\}$ and neuron $m\in\{0,1,\dots,M\}$,  $\sigma_{l,m}(z)$, is implemented as: 
  
  <p align="center">
    $\Large\sigma_{l,m}(z)=\sum\limits_{q=1}^{Q} F_{q,l}^{(m)}\cos\left(\frac{\pi(2q-1)(2\bar{z}+1)}{2N}\right),$
  </p>
  
  with $\bar{z}=\frac{N}{2}(z+1)$, and $F_{q,l}^{(m)}, q=1,\dots,Q$ is the set of DCT coefficients at neuron $m$ and layer $l$.
</details>

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

