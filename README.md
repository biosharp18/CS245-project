# CS245-project- Investigating the Robustness of poisoned RLHF feedback data

This repo contains the code needed to reproduce our project.

You will need to install conda to run this project:

```wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh```

Then install the environment from the yaml file:

```conda env create -f environment.yaml```

## Context-free grammar definition for output poisoning

See generate_toxicity.py and generate_toxicity_2.py for iterations.

## Training DPO model

See train_DPO.py

## Evaluating model, obtaining toxicity scores

See eval_model.py

