# BadHonnefTutorial

This repository contains the files for the tutorial sessions on tangent space methods for uniform matrix product stats, given at the 2020 school on Tensor Network based approaches to Quantum Many-Body Systems, held in Bad Honnef.

## Setup and installation

The tutorial is given in the form of python notebooks, which provide a detailed guide through the relevant concepts and algorithms, interspersed with checks and demonstrations. Reference implementations are also given in MATLAB.

The easiest way to get all the tools needed to open and run the notebooks is to install python via the anaconda distribution: https://docs.anaconda.com/anaconda/install/. This automatically comes with all packages needed for this tutorial.

If you already have a python version installed, you may install the jupyter notebook package seperately through <code>pip</code> by running
```console
pip install notebook
```

Once jupyter notebook is installed, clone this repository to a local folder of your choosing, open a terminal and navigate to this folder, and simply run
```console
jupyter notebook
```

For performing contractions of tensor networks, we have opted to use the python implementation of the <code>ncon</code> contractor (https://arxiv.org/abs/1402.0939), which can be found at https://github.com/mhauru/ncon. There are undoubtedly many contraction tools that work equally well and any of them may of course be used, but this one has a particularly intuituve and handy syntax. To install ncon, you may run
```console
pip install ncon
```
but we have also just included the source code for the ncon function in the tutorial folder for convenience.


## Contents

The tutorial consists of three parts:

#### 1. Matrix product states in the thermodynamic limit
This part is concerned with the basics of MPS in the thermodynamic limit: normalization, fixed points, algorithms for gauging an MPS, and computing expectation values.

#### 2. Finding ground states of local Hamiltonians
This part describes how to perform a variational search in the space of MPS to find the ground state of a local Hamiltonian. We start by considering a simple gradient based approach, and then introduce the vumps algorithm. We also briefly touch upon how excited states may be constructed using the result of the vumps ground state search.

#### 3. Transfer matrices and fixed points
In this part the vumps algorithm is extended to matrix product operators (MPOs), and it is then used to compute the partition function and magnetization for the 2d classical Ising model.


## Use of this tutorial
Each chapter provides a jupyter notebook (.ipynb) file with guided exercices on implementing the algorithms, as well as a solution notebook. Similar files are available for MATLAB.

The approach itself is very basic, where all algorithms are broken down and illustrated in digestible steps. The implementations are very simple: are no black boxes or fancy tricks involved, everything is built from the ground up. As a result, implementing all these steps from the start would most likely take longer than two tutorial sessions. The contents of the tutorial should therefore not just be seen as a task to complete, but just as much as a reference on how one would go about implementing these concepts. Please feel free to use this in any way you see fit.
