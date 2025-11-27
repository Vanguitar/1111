## One-dimensional latent diffusion with health-guided cross-attention mechanism

In the reverse diffusion process, a one-dimensional U-Net architecture is employed as the denoising network backbone. The network comprises three hierarchical levels with progressive downsampling and upsampling operations. At each level, self-attention mechanisms enable the network to capture long-range temporal dependencies within fault features. Crucially, cross-attention blocks are selectively integrated into the higher resolution levels of both the encoding and decoding paths.

### Fusion of source domain healthy features

The conditioning information for the training phase is derived from the fused healthy features of the source domain. Prior to entering the cross-attention mechanism, healthy signals from the source domain are aggregated across all samples to compute a representative condition signature. Specifically, the fused healthy feature for the source domain is computed as:

$$\mathbf{h}_s = \frac{1}{N_h} \sum_{i=1}^{N_h} \mathbf{H}_{s,i}$$

where $\mathbf{H}_{s,i}$ represents the healthy signal features from the source domain (e.g., at operating condition $s$), $N_h$ denotes the total number of healthy samples available from the source domain, and $\mathbf{h}_s$ is the averaged healthy feature representation that captures the operating-condition-specific perturbations and impulses characteristic of the source domain. This aggregated representation encodes the baseline vibration signature and environmental noise patterns unique to the source operating condition.

### Cross-attention mechanism with health-guided conditioning

The core innovation lies in the health-guided cross-attention mechanism, which injects domain-specific condition information into the feature synthesis process. During the training phase, the fused healthy features $\mathbf{h}_s$ are used as conditioning information within the cross-attention modules. These healthy features contain the specific perturbations and impulses induced by the source operating condition, such as vibration signatures due to particular load levels or rotational speeds. The cross-attention mechanism computes attention weights that link the noisy fault features (query) to the healthy condition features (key and value), enabling the model to learn how condition-specific perturbations should influence the denoising process.

Mathematically, the cross-attention operation is formulated as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where $Q \in \mathbb{R}^{L \times d_k}$ represents the query vectors derived from the current noisy features at sequence length $L$, $K, V \in \mathbb{R}^{1 \times d_k}$ are the key and value vectors computed from the conditioning healthy features (note that $\mathbf{h}_s$ is broadcast across the sequence dimension), and $d_k$ denotes the dimension of each attention head. The normalization by $\sqrt{d_k}$ ensures numerical stability. The resulting attention-weighted combination allows the model to selectively emphasize those aspects of the source condition that are most relevant to preserving domain-specific characteristics during feature generation.

### Delta conditioning for target domain adaptation

During the generation phase, the same cross-attention mechanism operates with different conditioning inputs. Instead of source domain healthy features, the model receives the difference between target domain and source domain healthy features. This delta conditioning approach is formulated as:

$$\mathbf{c}_{\Delta} = \mathbf{h}_t - \mathbf{h}_s$$

where $\mathbf{h}_t = \frac{1}{N_h'} \sum_{i=1}^{N_h'} \mathbf{H}_{t,i}$ represents the averaged healthy feature from the target domain (e.g., at operating condition $t$), $\mathbf{h}_s$ is the source domain healthy feature as previously defined, and $\mathbf{c}_{\Delta}$ represents the delta conditioning vector that captures the distributional shift between domains. 

This delta representation is particularly powerful because it explicitly encodes what distinguishes the target domain from the source without introducing actual faulty target-domain information (which is unavailable). The network learns to interpret $\mathbf{c}_{\Delta}$ as a transformation instruction: "apply these domain-specific adjustments to synthesize features that exhibit target-domain characteristics." By operating on the difference rather than absolute conditions, the mechanism becomes domain-agnostic and can generalize to multiple target domains, provided that healthy reference signals are available for those domains.

The selective placement of cross-attention modules is strategic. They are applied only at higher resolution levels of the network, where fine-grained temporal patterns are preserved. Lower resolution levels, which encode more abstract structural information, bypass cross-attention to focus on general fault characteristics that generalize across domains. This design prevents over-conditioning on domain-specific details while maintaining sufficient guidance to capture operating-condition-induced variations.


## Feature generation with Fault-Prior Self-Attention mechanism

The feature generation stage begins with random Gaussian noise, which is iteratively refined through the learned reverse diffusion process. To guide this synthesis toward fault-specific representations, a fault-prior self-attention mechanism is introduced, which constrains the generation to preserve fault-type characteristics throughout the denoising trajectory.

The self-attention blocks embedded within the U-Net architecture compute attention weights exclusively within the current feature representation. Unlike cross-attention which introduces external conditioning, self-attention enables the network to identify and reinforce internal structural patterns that distinguish different fault types. During generation, source-domain fault features serve as implicit prior constraints. The model has learned, through training on source fault data, which temporal and spectral patterns are inherent to each fault type, and this knowledge is encoded in the network weights and attention patterns.

The fault-prior self-attention mechanism operates through progressive refinement across diffusion steps. In early denoising steps, when noise is still substantial, the self-attention mechanism broadly identifies major structural features. As the process continues toward later steps, self-attention becomes increasingly fine-grained, refining subtle temporal patterns that distinguish fault types and severity levels. The mechanism prevents the generation process from drifting toward random features or misclassified fault types through step-by-step structural consistency enforcement.

Specifically, the forward process of generation proceeds as follows. Starting from Gaussian random noise $\mathbf{x}_T$ at timestep $T$, the model applies the reverse diffusion process for $T$ steps. At each step $t$, the network computes a prediction of the noise component using both self-attention (which constrains fault-type consistency) and the timestep embedding (which provides the denoising guidance for the current step). The prediction is used to refine the feature according to the equation:

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left[ \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_{\Delta}) \right]$$

where $\alpha_t$ is the noise schedule coefficient from the pre-defined diffusion schedule, $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$ is its cumulative product, $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_{\Delta})$ denotes the network's prediction of the noise component at step $t$, and $\mathbf{c}_{\Delta}$ represents the delta conditioning information (the difference between target and source domain healthy features as defined in the preceding section).

The self-attention mechanism is crucial for maintaining fault-type coherence. During training on source fault data, the network learns distinctive attention patterns for each fault type. These patterns capture which features correlate with specific fault classes. During generation, the same attention mechanisms reactivate these learned patterns, guiding the synthesis to produce features with characteristic fault signatures. For instance, ball fault features possess different spectral-temporal patterns compared to inner race faults, and the self-attention mechanism preserves these distinctions by preferentially amplifying fault-type-specific feature correlations.

The combination of self-attention for fault-type constraint and cross-attention for domain adaptation creates a balanced synthesis process. Cross-attention ensures that generated features align with the target operating condition, while self-attention ensures that despite condition changes, the fundamental fault characteristics are preserved. This dual-guidance strategy prevents feature collapse (where all features converge to averaging effects) and ensures that synthesized features remain fault-discriminative across domain shifts. The quality of generated features is maintained across the entire denoising trajectory through this integrated mechanism.


## Classifier generalization and novel domain diagnosis

Following feature generation, the synthesized target-domain fault features are combined with source-domain features to train a unified classifier. The training dataset consists of both the original source-domain features and the newly synthesized target-domain features, with balanced representation across all fault classes and severity levels.

A standard deep learning classifier, such as a convolutional neural network or fully connected network, is trained on this augmented dataset using cross-entropy loss. The inclusion of synthesized target-domain features exposes the classifier to examples that exhibit the statistical characteristics of the target domain while maintaining fault-class distinctions learned from the source domain. During testing on unseen target-domain samples, the classifier makes predictions based on the learned decision boundaries that have been informed by both source and generated target features.

The effectiveness of this approach depends critically on the quality and fidelity of synthesized features. The evaluation results presented in Section 4 demonstrate that synthesized features maintain strong consistency across multiple statistical metrics (Euclidean distance, Pearson correlation, MSE, and MAE), indicating that they capture both the magnitude and temporal structure of target-domain fault features. This consistency ensures that the classifier trained on synthesized features generalizes reliably to actual target-domain test samples.
