## Multi-view Feature Encoding and Fusion with Tiny Attention Mechanism

Industrial vibration signals exhibit complex characteristics that cannot be fully captured by analyzing a single representation. Different aspects of fault information are distributed across multiple domains, and a comprehensive understanding of fault mechanisms requires integrating information from these complementary perspectives. The proposed framework employs multi-view feature encoding to extract representations from three fundamental domains, which are then fused through a tiny attention mechanism that learns the relative importance of each view in an adaptive manner.

### 3.1.1 Multi-view Feature Encoding

The fundamental premise of the multi-view approach is that industrial fault signals contain heterogeneous information that manifests differently depending on how the signal is transformed and analyzed. The three chosen views—time domain, frequency domain, and time-frequency domain—represent distinct but complementary perspectives that together provide a complete characterization of the fault mechanism.

**Time-Domain Feature Encoding.** The time-domain representation preserves the temporal structure of the raw vibration signal, which carries critical information about the transient events and impulsive phenomena associated with bearing faults. For a given input signal $\mathbf{x} \in \mathbb{R}^{L}$ of length $L$, the time-domain features are extracted through a one-dimensional convolutional neural network encoder:

$$\mathbf{h}_t = \mathcal{F}_{\text{time}}(\mathbf{x}; \Theta_t)$$

where $\mathcal{F}_{\text{time}}(\cdot; \Theta_t)$ represents the time-domain CNN encoder with learnable parameters $\Theta_t$. The resulting features $\mathbf{h}_t \in \mathbb{R}^{C \times T}$ maintain the temporal ordering and local temporal correlations present in the original signal, with $C$ denoting the channel dimension and $T$ the temporal sequence length. The time-domain view is particularly valuable for detecting impulsive fault transients, as it captures the sudden amplitude changes and peak values that characterize bearing degradation.

**Frequency-Domain Feature Encoding and Theoretical Justification.** The frequency-domain representation obtained through the Fast Fourier Transform captures the global spectral composition of the signal, revealing the energy distribution across different frequencies. This transformation is motivated by fundamental signal processing principles rooted in the theory of bearing fault diagnosis. Industrial bearings exhibit characteristic failure modes, each producing distinctive frequency signatures. An outer race fault generates repetitive impulses with a frequency equal to the ball pass frequency of the outer race (BPFO), mathematically determined by:

$$f_{\text{BPFO}} = \frac{Z}{2}f_r\left(1 - \frac{d}{D}\cos\phi\right)$$

where $Z$ is the number of rolling elements, $f_r$ is the shaft rotation frequency, $d$ is the rolling element diameter, $D$ is the pitch diameter, and $\phi$ is the contact angle. Similarly, inner race faults produce energy at the ball pass frequency of the inner race (BPFI):

$$f_{\text{BPFI}} = \frac{Z}{2}f_r\left(1 + \frac{d}{D}\cos\phi\right)$$

These fault-specific frequencies are deterministically related to bearing geometry and are nearly impossible to identify reliably in the time domain, especially for early-stage faults with low signal-to-noise ratios. However, these characteristic frequencies and their harmonics become prominently visible in the frequency domain, where they appear as energy peaks at predictable locations.

The Fast Fourier Transform is defined as:

$$\mathbf{X}_{f} = \text{FFT}(\mathbf{x}) = \sum_{n=0}^{L-1} \mathbf{x}(n) e^{-j2\pi fn/L}$$

where $f$ ranges from $0$ to $L-1$, representing different frequency bins. The magnitude spectrum $\tilde{\mathbf{X}}_f = |\mathbf{X}_f| \in \mathbb{R}^{L/2+1}$ directly encodes the energy distribution across frequencies. The FFT provides several theoretical advantages. First, it decomposes the signal into an orthogonal basis of complex exponentials, ensuring that no information is lost from the frequency perspective—all components of the signal are captured with equal fidelity. Second, the FFT is mathematically complete: the inverse transform perfectly reconstructs the original signal, guaranteeing that the frequency domain contains all information present in the time domain. Third, from a computational perspective, the FFT has complexity $O(L \log L)$, making it practical for real-time industrial monitoring systems where computational efficiency is critical.

The frequency-domain features are extracted by processing the magnitude spectrum through a one-dimensional convolutional architecture:

$$\mathbf{h}_f = \mathcal{F}_{\text{freq}}(\tilde{\mathbf{X}}_f; \Theta_f)$$

where $\mathbf{h}_f \in \mathbb{R}^{C \times T}$ denotes the frequency-domain features and $\Theta_f$ contains the encoder parameters. By learning a CNN representation of the frequency spectrum, the model can automatically discover spectral patterns and relationships between frequency components that are indicative of specific fault types, without requiring manual specification of the characteristic frequencies.

**Time-Frequency-Domain Feature Encoding and Theoretical Justification.** While the frequency domain excels at revealing stationary spectral patterns, it fundamentally fails to provide temporal localization information. Consider a signal containing energy at frequency $f_0$ starting at time $t_1$ and ending at time $t_2$: the FFT reveals that energy exists at $f_0$ but provides no information about when it occurs. This temporal blindness becomes critical when dealing with non-stationary signals—signals whose spectral content changes over time—which are ubiquitous in industrial applications. As bearings degrade, the fault signatures typically evolve through different phases: incipient faults produce sparse, localized impulses; established faults produce more regular impulsive patterns; and advanced faults may exhibit continuous high-frequency oscillations. A representation that loses temporal information cannot distinguish these fault progression phases.

The time-frequency domain addresses this limitation through wavelet analysis, which simultaneously characterizes both temporal and spectral properties. The continuous wavelet transform is mathematically defined as:

$$\mathbf{W}_{s,\tau} = \int_{-\infty}^{+\infty} \mathbf{x}(t) \psi_{s,\tau}^{*}(t) dt$$

where $\psi_{s,\tau}(t) = \frac{1}{\sqrt{s}}\psi\left(\frac{t-\tau}{s}\right)$ is the scaled and shifted wavelet function, with scale parameter $s$ controlling the frequency resolution and time shift parameter $\tau$ controlling the temporal localization. This is fundamentally different from the FFT in a crucial way: wavelets use basis functions that are localized in both time and frequency domains, whereas sinusoids (the FFT basis) are delocalized in time.

The theoretical advantage of wavelets is grounded in the uncertainty principle, which states that perfect simultaneous localization in both time and frequency is impossible. More precisely, the product of time resolution (measured by the standard deviation of the wavelet in time) and frequency resolution (measured by the standard deviation in frequency) is bounded below by a constant:

$$\Delta t \cdot \Delta f \geq \frac{1}{4\pi}$$

Wavelets optimally balance this trade-off: they achieve some localization in both domains, whereas sinusoids achieve perfect frequency localization at the cost of zero time localization. This property is particularly valuable for bearing fault diagnosis, where fault signatures are inherently transient and localized in time. When a bearing develops a spall, the resulting vibration signature is a series of narrow-duration impulses at discrete times (when the rolling elements strike the spall) separated by periods of relative silence. A wavelet can precisely localize these impulses in time while simultaneously identifying the frequency content of each impulse through the scale parameter.

Moreover, wavelets provide multi-scale analysis capabilities. By varying the scale parameter $s$, the transform produces representations at multiple resolutions simultaneously: coarse scales (large $s$) capture low-frequency trends and overall envelope characteristics, while fine scales (small $s$) capture high-frequency details and localized transients. This multi-scale perspective aligns naturally with the hierarchical structure of bearing fault progression: faults exhibit characteristics at multiple time scales, from individual impulses (very fine scale) to overall envelope modulation (coarse scale).

The magnitude coefficients of the wavelet transform, $|\mathbf{W}_{s,\tau}|$, are arranged into a two-dimensional representation and resized to a standard dimension of $64 \times 64$:

$$\mathbf{M}_{w} = \text{Resize}(|\mathbf{W}_{s,\tau}|; 64 \times 64)$$

This resizing ensures uniform feature dimensions across different input signals of varying lengths. A two-dimensional convolutional encoder then processes this representation:

$$\mathbf{h}_w = \mathcal{F}_{\text{wavelet}}(\mathbf{M}_{w}; \Theta_w)$$

where $\mathbf{h}_w \in \mathbb{R}^{C \times T}$ denotes the time-frequency-domain features and $\Theta_w$ contains the encoder parameters. The 2D convolution naturally exploits the two-dimensional structure of the wavelet coefficient matrix, enabling the model to learn complex interactions between different scales and time positions.

**Complementarity and Information-Theoretic Justification.** From an information-theoretic perspective, the three views provide complementary information that together spans a more complete feature space than any single view alone. The information divergence between any two views can be expressed through the Kullback-Leibler divergence or other divergence measures, and ideally, the three views should have low correlation while collectively capturing all relevant fault information. The time domain captures local temporal structures (via local differencing in the CNN), the frequency domain captures global spectral patterns (via the orthogonal FFT basis), and the time-frequency domain captures localized spectral patterns (via the wavelet basis with joint time-frequency localization). Together, they form a frame in the signal processing sense—a redundant but complete basis for signal representation.

### 3.1.2 Fusion with Tiny Attention Mechanism

Having extracted complementary feature representations from three different domains, the central challenge is to combine them into a unified representation that preserves the strengths of each view while mitigating their individual limitations. A naive approach—such as simple concatenation or equal-weighted averaging—lacks the flexibility to adjust the importance of different views based on the input signal characteristics. Conversely, complex fusion mechanisms such as full multi-head attention networks, while powerful, are computationally expensive and may be unnecessarily complex for the fixed three-view scenario.

The tiny attention mechanism addresses this by learning lightweight, data-driven weights for each view that capture their relative importance. The mechanism is "tiny" in the sense that it uses minimal parameters and computational resources while maintaining full adaptability to the input data.

**Weight Computation via a Shallow Neural Network.** The weight computation is performed through a small multi-layer perceptron that processes the features at each temporal position. For the features at temporal position $i$, let $\mathcal{F}_{\text{concat},i}$ denote the concatenation of the feature vectors from the three domain views. The architecture of the weight network is deliberately simple, consisting of exactly two fully connected layers.

In the first layer, the concatenated features are projected to an intermediate representation of reduced dimensionality:

$$\mathbf{a}_i = \text{ReLU}(\mathcal{F}_{\text{concat},i} \mathbf{W}_1 + \mathbf{b}_1)$$

where $\mathbf{W}_1 \in \mathbb{R}^{C \times (C/4)}$ projects from dimension $C$ to dimension $C/4$, and $\mathbf{b}_1 \in \mathbb{R}^{C/4}$ is the corresponding bias. The ReLU activation function $\text{ReLU}(x) = \max(0, x)$ introduces crucial nonlinearity. Without this nonlinearity, the two fully connected layers would compose into a single linear transformation, capable of learning only linear importance functions. However, the relative importance of views depends on nonlinear relationships. For instance, the importance of the frequency-domain view versus time-domain view might depend on whether the high-frequency content exceeds a certain threshold, a nonlinear relationship that cannot be captured by linear transformations alone. The ReLU activation, while simple, provides sufficient expressive power for this task and is computationally efficient.

The dimensionality reduction from $C$ to $C/4$ serves multiple theoretical purposes. First, from a regularization perspective, this bottleneck forces the network to learn a compressed representation containing only essential information. Information bottleneck theory suggests that learning a compressed representation that retains sufficient statistics can improve generalization. The compression ratio of 4:1 is chosen based on empirical heuristics from network architecture design; alternative ratios (2:1, 8:1) could be used depending on specific problem characteristics. Second, this dimensional reduction dramatically decreases the parameter count: without it, the weight matrix $\mathbf{W}_1$ would be $C \times C$ with $C^2 = 4096$ parameters; with it, the combined parameter count for both layers is $C^2/4 + C = 1088$ parameters, a reduction by a factor of approximately 3.76. This efficiency is crucial for deployment scenarios.

In the second layer, the intermediate representation is projected to the view dimension, and a softmax function ensures that the weights form a valid probability distribution:

$$\mathbf{w}_i = \text{Softmax}(\mathbf{a}_i \mathbf{W}_2 + \mathbf{b}_2)$$

where $\mathbf{W}_2 \in \mathbb{R}^{(C/4) \times V}$ projects to $V=3$ view dimensions, and $\mathbf{b}_2 \in \mathbb{R}^{V}$ is the bias. The softmax function is critical to both the interpretability and theoretical soundness of the mechanism. Mathematically, the softmax function is given by:

$$w_i^{(v)} = \frac{\exp(a_i^{(v)})}{\sum_{v'=1}^{V} \exp(a_i^{(v')})}$$

where $a_i^{(v)}$ is the $v$-th component of the logits. The exponential function has the theoretical property of mapping unbounded real values to the positive domain, and the normalization ensures the weights sum to one, forming a valid probability distribution. This probabilistic interpretation is not merely mathematical convenience; it provides deep theoretical advantages. First, from a Bayesian perspective, the weights can be interpreted as posterior probabilities that each view contains the relevant information for that temporal position. Second, the constraint $\sum_v w_i^{(v)} = 1$ prevents degenerate solutions where all weights collapse to zero or become unboundedly large, ensuring stable gradient flow during backpropagation. Third, the softmax exponents amplify small differences between logit values—if the network produces slightly different values for different views, softmax magnifies these differences, making the mechanism sensitive to fine distinctions in view importance.

**Feature Fusion via Weighted Aggregation.** Once the importance weights for each view are computed, the final fused features are obtained through weighted aggregation. For each temporal position $i$, the fused feature is a weighted linear combination of the view-specific features:

$$\mathbf{h}_{\text{fused},i} = \sum_{v=1}^{V} w_i^{(v)} \cdot \mathbf{h}_{v,i}$$

where $\mathbf{h}_{v,i}$ denotes the feature vector from view $v$ at temporal position $i$. This aggregation is performed independently for each temporal position, enabling the mechanism to adapt the fusion to the local characteristics of the signal at different time instants.

The per-timestep weighting provides crucial flexibility grounded in signal processing theory. Different temporal regions of a bearing fault signal exhibit fundamentally different characteristics. In the early phase of fault development, the bearing surface develops a small spall, producing sharp, high-frequency impacts at regular intervals (determined by the bearing geometry and rotation speed). In this phase, time-domain and frequency-domain views both contribute valuable information, but the time-domain view excels at localizing the impact times while the frequency-domain view reveals the frequencies excited by impacts. As the fault progresses and the spall grows, the impacts become more frequent and the overall energy increases, shifting the balance of information content. In the late-stage advanced fault, the degradation may affect large bearing surfaces, producing continuous high-frequency oscillations in addition to impacts, further shifting the information balance. By computing weights at each time step, the mechanism can dynamically adjust which views contribute more to the fused representation, effectively capturing the time-varying nature of fault progression.

Following the weighted aggregation, an output projection layer is applied to produce the final fused features:

$$\mathbf{h}_{\text{fused}}^{\text{final}} = \text{Dropout}(\mathbf{h}_{\text{fused}} \mathbf{W}_O + \mathbf{b}_O)$$

where $\mathbf{W}_O \in \mathbb{R}^{C \times C}$ and $\mathbf{b}_O \in \mathbb{R}^{C}$ are the projection parameters. The output projection serves to linearly transform the aggregated features while maintaining the original feature dimensionality $C \times T$, facilitating seamless integration with downstream modules such as the latent diffusion model. The dropout regularization with a rate of 0.1 reduces overfitting by randomly zeroing a fraction of the features during training, forcing the network to learn redundant and robust representations that do not rely on specific channels.

**Comparison with Alternative Fusion Approaches.** The proposed tiny attention mechanism differs fundamentally from several alternative fusion strategies, each with distinct theoretical properties. Simple concatenation $\mathbf{h} = [\mathbf{h}_t; \mathbf{h}_f; \mathbf{h}_w]$ combines all features but assigns them equal importance regardless of the input signal or operating condition. This approach fails to recognize that the relevance of different views is input-dependent: for a signal with strong low-frequency content, the frequency-domain view is more informative; for a signal with sharp transients, the time-domain view is more informative. Equal-weighted averaging $\mathbf{h} = (\mathbf{h}_t + \mathbf{h}_f + \mathbf{h}_w)/3$ also lacks adaptability and, additionally, may lead to information cancellation if the views contain noisy or contradictory information.

Complex fusion mechanisms such as multi-head cross-attention with $H$ parallel attention heads can be expressed as:

$$\text{MultiHeadAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\mathbf{W}^O$$

where each head computes:

$$\text{head}_h = \text{Attention}(\mathbf{Q}\mathbf{W}_h^Q, \mathbf{K}\mathbf{W}_h^K, \mathbf{V}\mathbf{W}_h^V)$$

Such mechanisms can capture complex inter-view interactions and learn multiple complementary fusion strategies simultaneously through the different heads. However, they incur significant computational costs: each attention head requires computing similarity matrices between all pairs of temporal positions, resulting in $O(T^2 C H)$ complexity. For typical parameters ($T=64$, $C=64$, $H=8$), this becomes over 2 million floating-point operations per sample. In comparison, the tiny attention mechanism requires only $O(C^2) \approx 4096$ operations per sample, a reduction by more than 500$\times$. Moreover, multi-head attention operates on temporal sequences, learning relationships between different time positions, which is not directly applicable to the view fusion task where we want to learn relationships between views (a global property) rather than between time positions (local properties). The tiny attention mechanism aligns with the problem structure: computing importance scores for three views is inherently a small-scale problem that benefits from architectural simplicity.

### 3.1.3 Summary and Subsequent Processing

The multi-view feature encoding and fusion pipeline produces a comprehensive representation $\mathbf{h}_{\text{fused}}^{\text{final}} \in \mathbb{R}^{C \times T}$ that integrates temporal, spectral, and time-frequency information through learned, adaptive importance weights. This fused representation captures the complete information content distributed across the three domains, serving as input to subsequent domain generalization modules. The theoretical foundations of the approach—grounded in signal processing theory (FFT and wavelet analysis), regularization theory (information bottleneck), and Bayesian probability (softmax weights)—provide both interpretability and robust generalization properties for industrial fault diagnosis applications.
