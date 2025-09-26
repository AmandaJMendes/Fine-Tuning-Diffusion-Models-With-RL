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





\subsection{Hardware Setup}
All experiments were conducted on a single machine equipped with four NVIDIA T4 GPUs. To efficiently utilize the available computational resources and manage the memory requirements of diffusion model training, we employed data parallelism. This approach involves distributing the batch of input data across all available GPUs. Each GPU processes a subset of the batch independently, computes its local gradients, and then these gradients are aggregated and synchronized across all GPUs to perform a single optimizer step. This parallelization strategy significantly accelerates the training process by allowing simultaneous computation, making it feasible to fine-tune large diffusion models within a reasonable timeframe. However, the available 16GB memory per T4 GPU currently limits the maximum effective batch size we can use. For future experiments, the hardware setup will be extended to include GPUs with larger memory capacities, which will enable the use of larger batch sizes, further shorten training times, and allow for experiments to be conducted at a larger scale.

\subsection{Pretrained Model} 
We employ the publicly available diffusion model \texttt{google/ddpm-celebahq-256} \cite{pretrained_model} as the policy initialization for fine-tuning. This model was trained on the CelebA-HQ dataset \cite{celeba_hq}, a high-resolution subset of the CelebA dataset consisting of 30,000 facial images at \(1024 \times 1024\) resolution. CelebA-HQ preserves the attribute richness of the original CelebA dataset—such as age, gender, and hairstyle—while enhancing image quality. CelebA itself contains over 200,000 celebrity images. The dataset spans a diverse set of identities and includes variations in pose, lighting, and background, making it a widely adopted benchmark for generative modeling of faces.

Although the CelebA-HQ dataset provides images at \(1024 \times 1024\), the model was trained on downsampled versions at \(256 \times 256\), which is also the resolution used in our experiments. The model follows the DDPM framework and uses a U-Net architecture trained to predict the noise component \( \epsilon \) added during the forward diffusion process.

In this preliminary stage, we focus on unconditional generation and use the model in its original form, without any conditioning inputs. While our general formulation remains compatible with conditional diffusion models through the inclusion of a context variable \(c\), this simplification does not affect the mathematical structure of our approach.

Empirically, we observed that the model exhibits a strong demographic bias: when sampling unconditionally, the majority of generated images, approximately 78\%, depict women. This observation reflects imbalances in the underlying model pretrained on the CelebA-HQ dataset and highlights the importance of addressing such biases in downstream applications.


\subsection{Scheduler}
We employ a Denoising Diffusion Implicit Model (DDIM) scheduler \cite{ddim} to perform the reverse sampling process during training and inference. While the original DDPM formulation defines a purely stochastic Markov chain for this reverse process, DDIM introduces a more generalized framework that offers significant practical advantages within the diffusion modeling landscape. DDIM allows for a controllable amount of noise in each denoising step via the hyperparameter $\eta$. This flexibility is a key reason for its widespread adoption, as it enables faster sampling with fewer steps than DDPM, which is crucial for computational efficiency and practical application.

Specifically, setting $\eta=0$ yields a fully deterministic trajectory, making the reverse process non-Markovian, as the entire generation path is determined without intermediate stochasticity. Conversely, setting $\eta=1.0$ recovers the fully stochastic behavior of DDPM, effectively defining a Markovian reverse process where each step $x_{t-1}$ is sampled based solely on $x_t$.

In our experiments, we set $\eta = 1.0$. This specific choice is critical because it ensures that the reverse diffusion process behaves stochastically and Markovian, which is consistent with our formulation of the denoising trajectory as a multi-step Markov Decision Process for policy gradient reinforcement learning. Thus, while leveraging the broad utility and efficiency benefits of the DDIM framework, this configuration specifically aligns our practical implementation with the Markovian theoretical underpinnings of our policy gradient objective. The number of inference steps is fixed to $T=50$, and the corresponding variance schedule remains unchanged during fine-tuning. This fixed schedule is crucial for computing log-likelihoods consistently, as required by the policy gradient estimator and the PPO objective.

\subsection{Reward Functions}
\label{sec:reward_functions_used}
As a case study, we apply our method to counteract the demographic bias exhibited by the pretrained diffusion model, specifically its tendency to generate a disproportionate number of female faces. Our goal is to guide the model toward producing more male-presenting images, without compromising the realism or overall image quality. To achieve this, we define a composite reward function that integrates multiple perceptual and attribute-based signals. Given a batch of final generated images \( x_0 \), we compute the following components:
\begin{itemize}
    \item \textbf{ImageReward (IR):} A reward model trained to reflect human aesthetic preferences, as described in Section~\ref{sec:reward_models}. Since this model expects a text prompt as input, we use: \textit{``a natural, high-quality portrait photograph of a person with realistic facial features, normal hair color, natural expression, and clean background''}.
    \item \textbf{Gender Reward:} This binary reward component is designed to mitigate demographic bias and encourage the generation of male-presenting images. It leverages an external pre-trained image classification pipeline \cite{gender_model} to predict the probability of an image being male. If the predicted male probability is $\ge 0.8$ or $\le 0.2$ (indicating high confidence), that probability is used as the score; otherwise, the score is set to 0.0 (for ambiguous or low-confidence predictions). This continuous score is then binarized: images with a male probability greater than or equal to 0.8 receive a reward of 1, and 0 otherwise. This mechanism directly steers the model to shift the gender distribution of generated outputs
    \item \textbf{Aesthetic Score:} A continuous score from the LAION Aesthetic Predictor \cite{laion_repo}, discussed in Section~\ref{sec:reward_models}. This score is recorded for evaluation purposes but not directly used in the training reward.
\end{itemize}

The total reward used for optimization is computed as:
\[
r(x_0) = \text{IR}(x_0) + 2 \cdot \text{GenderReward}(x_0),
\]
where \text{IR} is the ImageReward score and \text{GenderReward} is the binary reward signal described above. All component scores are logged for evaluation and ablation.


\subsection{Summary of Configuration} 
To provide a concise reference, Table~\ref{tab:hyperparams_structured} summarizes the key hyperparameters and configuration details used in the experiments. For brevity, we refer to a single \textit{iteration} as one full cycle of training that includes: (1) generating a fixed batch of new samples, and (2) performing multiple PPO update epochs using those samples. An \textit{optimizer step} refers to a single parameter update performed by the optimizer based on a minibatch of trajectories from the current batch.


\begin{table}[htb]
  \centering
  \caption{Hyperparameters and configuration details used in the preliminary experiments}
  \label{tab:hyperparams_structured}
  \begin{tabular}{ll}
    \toprule
    \multicolumn{2}{l}{\textbf{Hardware Setup}} \\
    \midrule
    Number of GPUs & 4 (NVIDIA T4) \\
    Batch Size per GPU & 5 \\
    Effective Batch Size & 20 \\
    \midrule
    \multicolumn{2}{l}{\textbf{Model}} \\
    \midrule
    Pretrained Model & \texttt{google/ddpm-celebahq-256} \\
    Architecture & U-Net, 114M parameters \\
    Dataset & CelebA-HQ (256×256 resolution) \\
    Generation Type & Unconditional \\
    \midrule
    \multicolumn{2}{l}{\textbf{Scheduler}} \\
    \midrule
    Scheduler Type & DDIM \\
    DDIM Noise Control ($\eta$) & 1.0 (fully stochastic) \\
    Inference Timesteps ($T$) & 50 \\
    Noise Schedule & Linear \\
    $\beta_{\text{start}}$ & 0.0001 \\
    $\beta_{\text{end}}$ & 0.02 \\
    \midrule
    \multicolumn{2}{l}{\textbf{Reward Configuration}} \\
    \midrule
    Train & ImageReward and Gender Classification Score \\
    Evaluation & Aesthetic Score \\
    Reward Formula & $r(x_0) = \text{IR}(x_0) + 2 \cdot \text{GenderReward}(x_0)$ \\
    \midrule
    \multicolumn{2}{l}{\textbf{Optimization Settings}} \\
    \midrule
    Optimizer & AdamW \\
    Learning Rate & $1 \times 10^{-6}$ \\
    Samples per Iteration & 100 \\
    PPO epochs per Iteration & 2 \\
    Optimizer Steps per Iteration & 10 \\
    \bottomrule
  \end{tabular}
  %\fonte{Produced by the author.}
\end{table}

\section{PPO: Clipped Objective and Advantage Normalization}

PPO is utilized to improve the stability and sample efficiency of the policy gradient updates. It achieves this by constraining the magnitude of policy changes during training and allowing for off-policy updates.


PPO constrains the magnitude of policy updates via a probability ratio that compares the new and old policies:

\begin{equation}
r_t(\theta) = \frac{\pi_\theta(x_{t-1} \mid x_t, c)}{\pi_{\theta_{\text{old}}}(x_{t-1} \mid x_t, c)}
\label{eq:importance-ratio}
\end{equation}

This ratio measures the change in likelihood of each action (i.e., denoising step) under the updated policy relative to the one that generated the data. To prevent excessively large updates that could degrade performance, PPO applies a clipping function to this ratio, limiting it to a small interval around 1.

To further stabilize training, we replace the raw reward with a normalized advantage:

\begin{equation}
A(x_0, c) = \frac{r(x_0, c) - \mu}{\sigma}, 
\label{eq:advantage-definition}
\end{equation}
where \( \mu \) and \( \sigma \) are the running mean and standard deviation of observed rewards. This standardization ensures that reward magnitudes are well-scaled across training, which helps mitigate exploding or vanishing gradients.



To evaluate the practical effects of fine-tuning on model behavior, we measured the proportion of male-presenting images generated throughout training using a fixed classification threshold (as described in Section~\ref{sec:reward_functions_used}). Figure~\ref{fig:binary_gender_ratio} shows the evolution of this ratio across training iterations for different sampling strategies. All runs begin with approximately 20 to 30\% male-presenting samples, reflecting the gender imbalance inherent to the pretrained model. 

We further illustrate this effect in Figure~\ref{fig:sample_evolution}, which shows samples generated from a fixed random seed at different stages of training. By holding the seed constant, both the initial noise and the sequence of stochastic sampling decisions remain fixed across checkpoints, ensuring that differences in the outputs reflect only the effect of model updates.  The visual progression highlights how fine-tuning drive changes in the outputs: initial generations reflect the bias of the pretrained model, while later ones increasingly depict male-presenting individuals with improved visual quality. This qualitative trajectory reinforces the effectiveness of reward-aligned fine-tuning.
