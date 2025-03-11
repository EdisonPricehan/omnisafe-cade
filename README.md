# CADE: Constrained Actor Dynamics Estimator for Safe Reinforcement Learning


## Overview
CADE is a **model-based Safe Reinforcement Learning (SafeRL)** framework designed to address vision-driven autonomous river following in challenging environments where GPS signals are unreliable.
This repository implements the **Marginal Gain Advantage Estimation (MGAE)** method, the **Semantic Dynamics Model (SDM)** for interpretable state prediction, and the **CADE** architecture, which integrates safety-aware model-based RL to solve **partially observable Constrained Submodular Markov Decision Processes (PO-CSMDPs)**.
Such environments include [CliffCircular-v1](https://github.com/EdisonPricehan/CliffCircular) and [Safe Riverine Environment](https://github.com/EdisonPricehan/ml-agents-river), which are both PyPI packages.
The whole repo is developed on top of [OmniSafe](https://github.com/PKU-Alignment/omnisafe). Thanks to their contributions to SafeRL.

## Key Contributions

- **MGAE**

MGAE is designed to handle non-Markovian rewards in SMDPs, where future rewards depend on historical trajectory rather than just the current state.
Unlike GAE, which bootstraps future rewards using a value critic, MGAE looks backward by leveraging a recurrent reward estimator to approximate immediate rewards.
This avoids bias from value function errors and ensures advantage estimation aligns with the cumulative nature of submodular rewards.
By focusing on marginal historical gains rather than future predictions (state value), MGAE provides a more stable and accurate policy update in environments where rewards depend on past exploration.

![mgae-gae-comp](images/mgae-gae-comp.png)

- **SDM**

Instead of learning vision dynamics in latent space, SDM estimates the homography transformation between consecutive semantic observations.
The semantic observation in Safe Riverine Environment is the patchified water mask from drone view.

![patchification](images/patchified_image.png)

The comparison of SDM with other vision dynamics models (e.g., Lagent Dynamics Model) is graphically shown below.

![vdm-comp](images/vision_dynamics_models_pred_compressed.mp4)

<video width="320" height="240" controls>
  <source src="images/vision_dynamics_models_pred_compressed.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

- CADE

CADE is built on top of the policy gradient method First Order Constrained Optimization in Policy Space ([FOCOPS](https://proceedings.neurips.cc/paper_files/paper/2020/hash/af5d5ef24881f3c3049a7b9bfe74d58b-Abstract.html)), which uses the Lagrangian multiplier to balance reward advantage and cost advantage in policy update, and integrates the KL divergence loss in the same policy loss.
But the differences are:
1. The actor and reward estimator share the same recurrent network (GRU).
2. No critics, just a reward estimator and a cost estimator that estimates the immediate reward and immediate cost.
3. The reward advantage is calculated by MGAE.
4. The cost advantage is defined as the discounted cumulative sum of predicted costs (by SDM and actor) in a short horizon, then transformed by sigmoid function.
5. The training goes episode by episode, instead of batch by batch.
6. Safety can also be injected during inference phase by the **safety layer with cost planning**, which also uses SDM and actor for planning.

The diagram of CADE's 4 components is shown below.

![diagram](images/diagram-v3.png)

The computational graph of CADE shows the forward pass and backpropagation pass, and the inputs and outputs of all components.

![cg](images/cade-computation-graph.png)


