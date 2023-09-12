# Symbolic AI with Inductive Logic Neural Network

This is the Python implementation of the Probabilitical T-norm mathematical architecture, wIth three different ways of parametrization  functions (Interpolation ratio, Pearson's r and Conditional ratio based).

Except for the original 3D and 2D visualiztions, all the three parametrised functions are visualized in two further approaches, beta distributiona and piecewise log, based on the concept of Quasi Maximum Likelihood.

Then the class module of T-norm has been verify in unit test. A toy expeiment with two latent variables has been developed.


## Requirements

To install the various python dependencies:

```
pip install -r requirements.txt
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Output

The code outputs the visualization images and evaluation of T-norm functions on the datasets provided. The metrics used for evluation are log loss and accuracy. The expriments' dataset and results are recorded in log file. 


## Code Structure

The strcture of the code is as follows:

```
Code
├── Model
│    ├── tnorm activation
│        ├── tnorm binary
│        └── tnorm multivariable
│    ├── tnorm latent variables
│    ├── tnorm generate data
│    └── unit test
├── Visualization
│    ├── plot norms
│    ├── plot quasiml
│    └── plot compose
```
