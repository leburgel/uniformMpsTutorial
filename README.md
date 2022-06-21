# uniformMpsTutorial

This repository contains a tutorial on tangent space methods for uniform matrix product states, originally written to be taught at the [2020 school on Tensor Network based approaches to Quantum Many-Body Systems](http://quantumtensor.pks.mpg.de/index.php/schools/2020-school/) held in Bad Honnef, Germany. The tutorials are based on the lecture notes ['Tangent-space methods for uniform matrix product states'](https://doi.org/10.21468/SciPostPhysLectNotes.7) by Laurens Vanderstraeten, Jutho Haegeman and Frank Verstraete.

## Setup and installation

The tutorial is given in the form of IPython Notebooks, which provide a detailed guide through the relevant concepts and algorithms, interspersed with checks and demonstrations. Reference implementations are also given in MATLAB.

The easiest way to get all of the tools needed to open and run the notebooks is to install Python via the anaconda distribution: https://docs.anaconda.com/anaconda/install/. This automatically comes with all packages needed for this tutorial.

If you already have a python version installed, you may [install Jupyter](https://jupyter.org/install) seperately via the [Python Package Index](https://pypi.org/) by running
```console
pip install notebook
```

Once jupyter notebook is installed, clone this repository to a local folder of your choosing, open a terminal and navigate to this folder, and simply run
```console
jupyter notebook
```

For performing contractions of tensor networks, we have opted to use the Python implementation of the [<code>ncon</code> contractor](https://arxiv.org/abs/1402.0939), which can be found at https://github.com/mhauru/ncon. There are undoubtedly many contraction tools that work equally well and any of them may of course be used, but this one has a particularly intuituve and handy syntax. To install ncon, you may run
```console
pip install ncon
```
but we have also included the source code for the ncon function in the tutorial folder for convenience.


## Contents

The tutorial consists of three parts:

#### 1. Matrix product states in the thermodynamic limit
This part is concerned with the basics of MPS in the thermodynamic limit: normalization, fixed points, algorithms for gauging an MPS, and computing expectation values.

#### 2. Finding ground states of local Hamiltonians
This part describes how to perform a variational search in the space of MPS to find the ground state of a local Hamiltonian. We start by considering a simple gradient based approach, and then introduce the [VUMPS algorithm](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.045145). We also briefly touch upon how excited states may be constructed using the result of the VUMPS ground state search. The algorithms are demonstrated for the case of the one-dimensional quantum spin-1 Heisenberg model.

#### 3. Transfer matrices and fixed points
In this part the VUMPS algorithm is extended to transfer matrices in the form of matrix product operators (MPOs), which can then be used to contract infinite two-dimensional tensor networks. The algorithm is demonstrated for the case of the classical two-dimensional Ising model, and is for example used to compute its partition function and evaluate the expectation value of the magnetization.


## Use of this tutorial
Each chapter provides a notebook (.ipynb) file with guided exercices on implementing the algorithms, as well as a solution notebook. Similar files are available for MATLAB. The approach itself is very basic, where all algorithms are broken down and illustrated in digestible steps. The implementations are very simple: there are no black boxes or fancy tricks involved, everything is built from the ground up. While these tutorials were originally intended to be taught at a school on tensor networks, they now serve as a general reference on uniform MPS and how one would go about implementing these concepts.
