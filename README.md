# Learning to Play General-Sum Games Against Multiple Boundedly Rational Agents

This is the code that accompanies the paper *Learning to Play General-Sum Games Against Multiple Boundedly Rational Agents*, published in AAAI 2023. If you use this work, please cite:

```bibtex
@inproceedings{ztxz22,
    title = {{Learning} {to} {Play} {General-Sum} {Games} {Against} {Multiple} {Boundedly} {Rational} {Agents}},
    arxiv = {2106.05492},
    url = {https://arxiv.org/abs/2106.05492},
    booktitle = {In Proceedings of the 37th AAAI Conference on Artificial Intelligence.},
    author = {Zhao, Eric and Trott, Alexander R. and Xiong, Caiming and Zheng, Stephan},
    month = {Feb},
    year = {2023},
    abstract = {We study the problem of training a principal in a multi-agent general-sum game using reinforcement learning (RL). Learning a robust principal policy requires anticipating the worst possible strategic responses of other agents, which is generally NP-hard. However, we show that no-regret dynamics can identify these worst-case responses in poly-time in smooth games. We propose a framework that uses this policy evaluation method for efficiently learning a robust principal policy using RL. This framework can be extended to provide robustness to boundedly rational agents too. Our motivating application is automated mechanism design: we empirically demonstrate our framework learns robust mechanisms in both matrix games and complex spatiotemporal games. In particular, we learn a dynamic tax policy that improves the welfare of a simulated trade-and-barter economy by 15\%, even when facing previously unseen boundedly rational RL taxpayers.},
}
```

## Overview

Throughout this repository, ERMAS ("Epsilon-Robust Multi-Agent Simulations") will refer to Algorithm 3 described in the paper.

The primary script is `main.py`, which can be run as:

```python
python3 main.py --run-dir ./{YOUR RUN DIRECTORY} --use-wandb`
```

Your run directory must have a `config.yaml` file inside of it meeting the traditional Ray Trainer specifications.
The `yamls/` directory includes all configurations necessary for reproducing the paper's experiments (see next section).

Some notes:

- This repository builds on Ray v0.8.7 and requires a custom Ray installation, described by the patch file under `ray_0.8.7-patch/`.
- This repository implements several environments compatible with Ray/RLLib's multi-agent learning API under `envs/`.
- The `BimatrixEnvWrapper` implements the Sequential Bimatrix Game experiments summarized in Figure 2.
- The `CoopBimatrixEnvWrapper3` class implements the N-Matrix game experiments summarized in Figure 3.
- The `AIEnvWrapper` implements the AI Economist Tax Policy experiments summarized in Figure 4.
- Utility files in this repository include `tf_models.py`, which implements Tensorflow networks (e.g., CNNs) for our agents to use, and various `utils/` files for saving/logging.

## Reproducing experiments from the paper

- Figure 2 can be replicated from the configurations in `bimatrix`. 
- Figure 3 can be replicated from the configurations in `coop_bimatrix_3`. 
- Figure 4 can be replicated from the configurations in `aie`. 
- Figure 5 can be replicated from the configurations in `coop_bimatrix_3_fixed`. 

Before these configurations can be used, several placeholder values must be updated as follows.

To run any experiment:

- You will occasionally find a configuration entry `SWEEP` mapping to a list of entries (usually floats/integers). This indicates that you must replace the SWEEP value with ONE of the entries in the list. Technically, to replicate all experiments in the paper, you will need to re-run the experiment for every possible combination of SWEEP values.

To run AI Economist experiments:

- Download the AI Economist agent checkpoint weights for agents trained under the Free Market (fmarket), Saez (saez), and US Federal (usfederal) tax policies. Configuration files with prefix `aie/eval_` or `aie/replicate_` may contain one of the following placeholders: `AIE_USFEDERAL_AGENT_WEIGHTS`, `AIE_SAEZ_AGENT_WEIGHTS`, `AIE_FMARKET_AGENT_WEIGHTS`. Replace these placeholders with the path to the corresponding agent checkpoint weights.
- First run the experiment described by configuration `aie/replicate_dplanner.yaml`. Upon completion, in all configuration files with prefix `aie/train_` and `aie/eval_`, replace the placeholder `YOUR_REPLICATION_CKPTS` with a path to the `replicate_dplanner` experiment's checkpoint directory.

To run an evaluation experiment (prefix `*/eval_`):

- First run the corresponding training experiment `*/train_`. Then, replace the evaluation configuration's placeholder `YOUR_EXPERIMENT_CKPTS` with a path to your training experiment's checkpoint directory.

## Installation & Usage

Install all requirements with

```
pip install -r requirements.txt
```

Then, sign into Weights and Biases with your API token: 

```
wandb login $WANDB_API_TOKEN
```

You will now need to install a specialized copy of Ray.

First run 

```
pip install ray==0.8.7
```

Then clone the [Ray repository](https://github.com/ray-project/ray), enter its root directory, and apply the `ray_0.8.7-patch` via 

```
git clone https://github.com/ray-project/ray
cd ray
git checkout releases/0.8.7
cp <PATH_TO_THIS_REPO>/ray_0.8.7-patch/ray_0.8.7_mods.patch .
git apply ray_0.8.7_mods.patch
python python/ray/setup-dev.py --yes
```

Create a run directory, e.g. `run`, and add your experiment configuration file at `run/config.yaml`.

Then call 

```
python main.py --run-dir ./run --use-wandb
```

Your experiments will be monitorable live on Weights and Biases, and logs and model weights will be regularly saved to the run directory.