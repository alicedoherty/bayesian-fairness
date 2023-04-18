## Individual Fairness in Bayesian Neural Networks

This repository contains all of the files to reproduce the experiments reported in the paper "Individual Fairness in Bayesian Neural Networks." The contents description and instructions to reproduce the results are listed below.

### Repository Resources

The following files are not run directly, but provide the following resources for our runs:

* `deepbayes/*` - A copy of the Bayesian deep learning framework used for inference
* `PaperOutputs/*` - A copy of all the exact numbers used to generate our paper figures
* `folk_utils.py` - A file to handle the folktables dataset
* `indiv_utils.py` - A file for learning IF metrics, and projecting onto ellipses 
* `deepensemble.py` - A file that uses the deepbayes interface to train a deep ensemble

### Reproducibility Instructions

The following files can be run directly to reproduce our results:

* `samples_attack_effect_learning.py` - Trains the networks we will study
* `samples_attack_effect_attacking.py` - Attacks the trained networks using FGSM
* `fairpgd_for_bnns.py` - Trains and evaluates a BNN against the proposed fairPGD attack
* `fairpgd_for_dnns.py` - Trains and evaluates a DNN against the proposed fairPGD attack
