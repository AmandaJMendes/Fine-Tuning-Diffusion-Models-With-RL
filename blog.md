Diffusion probabilistic models \cite{ddpm} have become a dominant framework for generative modeling, achieving state-of-the-art results in image synthesis \cite{beat_gans, stable_diffusion} and enabling advances in video \cite{video_diffusion_models}, 3D modeling \cite{3d_diffusion}, biology \cite{diffusion_biology}, and language \cite{d1, diffusion_lm}. Despite their success, these models are typically trained with maximum likelihood on massive uncurated datasets, which ensures fidelity to the data distribution but overlooks task-specific objectives such as aesthetic quality, alignment, or fairness \cite{no_code, pinterest}. 

To adapt diffusion models to downstream goals, a particularly promising approach is policy gradient fine-tuning \cite{ddpo, dpok, pinterest, no_code}, which frames the denoising trajectory as a multi-step decision-making process and enables direct optimization of non-differentiable black-box reward functions. Building on earlier reward-based approaches \cite{raft, rwr, draft, imagereward, d3po, diffusion_dpo, diffusion_kto}, policy gradient fine-tuning has achieved strong results in areas such as human preference modeling, prompt-image alignment, and fairness.

To enable this formulation within reinforcement learning framework, the reverse diffusion process is reframed as a multi-step Markov Decision Process (MDP) following \citet{ddpo, dpok, pinterest, no_code, no_pub}, with components defined as follows:

\[
s_t \equiv (c, t, x_t), \quad 
a_t \equiv x_{t-1}, \quad 
\pi_\theta(a_t \mid s_t) \equiv p_\theta(x_{t-1} \mid x_t, c),
\]
\[
P(s_{t+1} \mid s_t, a_t) \equiv (\delta_c, \delta_{t-1}, \delta_{a_t}), \quad 
R(s_t, a_t) \equiv 
\begin{cases}
r(x_0, c), & t=0 \\
0, & t>0
\end{cases}, \quad
P(s_0) \equiv (c, T, \mathcal{N}(0, I)), \quad 
\]

Here, $s_t$ denotes the state at time step $t$ consisting of the current noisy image $x_t$ and context $c$, $a_t$ is the action corresponding to predicting the denoised image $x_{t-1}$, $\pi_\theta$ is the reverse diffusion step parameterized by $\theta$, $P$ describes the deterministic transition to the next state, and $R$ gives a sparse reward only when the final image $x_0$ is produced. The process starts from the fully-noised image $x_T$ at $s_0$ and progresses backward toward the clean image $x_0$ at $s_T$.

Within this MDP, the policy gradient for maximizing te expcted reward follows the REINFORCE algorithm \cite{rl_book, reinforce}. By treating the log-probability of the trajectory as the object of differentiation, the gradient can be estimated using Monte Carlo samples weighted by their rewards:


\begin{equation}
\nabla_\theta J(\theta) = \mathbb{E}_{c \sim p(c), \tau \sim p_\theta(\tau \mid c)} \left[ r(x_0, c) \sum_{t=1}^{T} \nabla_\theta \log p_\theta(x_{t-1} \mid x_t, c) \right],  
\label{eq:reinforce_diffusion}
\end{equation}


where $\tau = (x_T, x_{T-1}, \ldots, x_0)$ denotes the trajectory of noisy images produced by the reverse diffusion process under policy $p_\theta(\cdot \mid \cdot, c)$. This expression is unbiased and straightforward to estimate via Monte Carlo sampling. Given a batch of \( N \) trajectories \( \{\tau^{(i)}\}_{i=1}^{N} \), each conditioned on context \( c^{(i)} \sim p(c) \), the gradient can be approximated as:

\begin{equation}
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} r(x_0^{(i)}, c^{(i)}) \sum_{t=1}^{T} \nabla_\theta \log p_\theta(x_{t-1}^{(i)} \mid x_t^{(i)}, c^{(i)})    
\end{equation}




