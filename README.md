Background
Diffusion models
DDPM (Discrete diffusion)
Forward and backward pass
Training loss (lower bound, etc)
Sampling 
MDP and RL
Policy gradient methods


Problem statement
Intro: Policy gradient method rely on exact likelihoods. In diffusion models, the exact likelihood is intractable 
So we frame the denoising as a MDP
At each step, we can use the exact likelihoods at each denoising step in place of the approximate likelihoods induced by a full denoising process, and use policy gradient methods (RL) for parameterizing/optimizing the MDP policy
Also, we consider that in policy gradient methods we can sample some steps along the episodes and keep an unbiased estimator, need to prove that. In the formulation, leave the sampler generic, this is part of the statement
The goal is to determine the sampler, it is generic, so it could be something that samples all the timesteps for instance. 
The final problem statement considers the reinforce loss but with the generic sampler included


Related Works
Finetuning diffusion models
Other methods that are not reward-based
Using reward-based finetuning
SFT
Direct backprop
Policy gradient
Non-uniform timestep 
Weighting
Sampling
Fixed distribution
Adaptative distribution 




