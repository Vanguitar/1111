## Multi-view Feature Encoding and Fusion with Tiny Attention Mechanism

Industrial vibration signals contain abundant information distributed across multiple domains. To capture comprehensive fault characteristics from diverse perspectives, the proposed framework employs multi-view feature encoding to extract features from three complementary domains, which are then fused through a tiny attention mechanism. This section describes the multi-view feature encoding strategy and the subsequent fusion process.

### 3.1.1 Multi-view Feature Encoding

The multi-view feature encoding process extracts feature representations from three distinct domains: time domain, frequency domain, and time-frequency domain. Each view provides a unique perspective on the input signal and captures different aspects of the fault characteristics.

For the time-domain view, the raw vibration signal $\mathbf{x} \in \mathbb{R}^{L}$ of length $L$ is processed through a one-dimensional convolutional neural network encoder:

$$\mathbf{h}_t = \mathcal{F}_{\text{time}}(\mathbf{x}; \Theta_t)$$

where $\mathcal{F}_{\text{time}}(\cdot; \Theta_t)$ represents the time-domain CNN encoder with learnable parameters $\Theta_t$. The resulting time-domain features $\mathbf{h}_t \in \mathbb{R}^{C \times T}$ maintain the temporal structure of the original signal, with $C$ denoting the feature channel dimension and $T$ the temporal sequence length.

For the frequency-domain view, the frequency spectrum is obtained by applying the Fast Fourier Transform to the input signal:

$$\mathbf{X}_{f} = \text{FFT}(\mathbf{x})$$

The magnitude spectrum is extracted as $\tilde{\mathbf{X}}_f = |\mathbf{X}_f| \in \mathbb{R}^{L/2+1}$, which captures the spectral content of the signal. This magnitude spectrum is then processed through an identical one-dimensional convolutional architecture:

$$\mathbf{h}_f = \mathcal{F}_{\text{freq}}(\tilde{\mathbf{X}}_f; \Theta_f)$$

where $\mathbf{h}_f \in \mathbb{R}^{C \times T}$ denotes the frequency-domain features and $\Theta_f$ contains the corresponding encoder parameters.

For the time-frequency-domain view, the continuous wavelet transform is employed to capture multi-scale temporal-spectral information:

$$\mathbf{W}_{s,\tau} = \int_{-\infty}^{+\infty} \mathbf{x}(t) \psi_{s,\tau}^{*}(t) dt$$

where $\psi_{s,\tau}(t) = \frac{1}{\sqrt{s}}\psi\left(\frac{t-\tau}{s}\right)$ is the wavelet function with scale parameter $s$ and time shift $\tau$. The magnitude coefficients are arranged into a two-dimensional matrix and resized to a standard dimension of $64 \times 64$:

$$\mathbf{M}_{w} = \text{Resize}(|\mathbf{W}_{s,\tau}|; 64 \times 64)$$

A two-dimensional convolutional encoder processes this time-frequency representation:

$$\mathbf{h}_w = \mathcal{F}_{\text{wavelet}}(\mathbf{M}_{w}; \Theta_w)$$

yielding the time-frequency-domain features $\mathbf{h}_w \in \mathbb{R}^{C \times T}$, where $\Theta_w$ denotes the encoder parameters.

The multi-view feature encoding process produces three heterogeneous feature representations, each capturing complementary information from a different domain perspective. The time-domain features preserve temporal dynamics, the frequency-domain features reveal spectral characteristics, and the time-frequency-domain features provide multi-scale temporal-spectral patterns. Integrating these three views into a unified representation requires a fusion mechanism that can adaptively weight the contributions of each view.

### 3.1.2 Fusion with Tiny Attention Mechanism

The three encoded feature representations are fused through a tiny attention mechanism that computes view-level attention weights and combines the features accordingly. Unlike conventional fusion approaches that employ fixed rules or complex interaction patterns, the tiny attention mechanism learns data-driven weights for each view in a lightweight and interpretable manner.

The tiny attention mechanism operates by first gathering the three view features and computing importance weights through a shallow neural network. The three encoded features $\mathbf{h}_t$, $\mathbf{h}_f$, and $\mathbf{h}_w$ are stacked together to form a multi-view feature representation. At each temporal position, the mechanism computes a weight for each view that indicates its relative importance in the final fused representation.

**Weight Computation and Aggregation.** For each temporal position in the feature sequence, the tiny attention mechanism computes three scalar weights, one for each view. Let the features at temporal position $i$ be denoted as the column vectors extracted from the three domain views. The weight computation is performed through a small multi-layer perceptron that processes the stacked view features. Specifically, the mechanism first applies a linear transformation that projects the concatenated feature vector from dimension $C$ to a reduced dimension of $C/4$, followed by a ReLU activation function to introduce nonlinearity:

$$\mathbf{a}_i = \text{ReLU}(\mathcal{F}_{\text{concat},i} \mathbf{W}_1 + \mathbf{b}_1)$$

where $\mathcal{F}_{\text{concat},i}$ represents the concatenation of the features from all three views at temporal position $i$, $\mathbf{W}_1 \in \mathbb{R}^{C \times (C/4)}$ is the first weight matrix that reduces the dimensionality, $\mathbf{b}_1 \in \mathbb{R}^{C/4}$ is the corresponding bias vector, and $\mathbf{a}_i \in \mathbb{R}^{C/4}$ is the intermediate activation. This dimensionality reduction serves to decrease computational complexity while maintaining sufficient representational capacity to capture the essential information needed for computing view weights.

The second layer of the weight network projects the intermediate representation from dimension $C/4$ to the view dimension, which is three in this case. This projection is followed by a softmax function to ensure that the computed weights form a valid probability distribution over the views:

$$\mathbf{w}_i = \text{Softmax}(\mathbf{a}_i \mathbf{W}_2 + \mathbf{b}_2)$$

where $\mathbf{W}_2 \in \mathbb{R}^{(C/4) \times V}$ is the second weight matrix that projects to $V=3$ view dimensions, $\mathbf{b}_2 \in \mathbb{R}^{V}$ is the bias vector, and $\mathbf{w}_i \in \mathbb{R}^{V}$ are the computed weights. The softmax normalization ensures that $\sum_{v=1}^{V} w_i^{(v)} = 1$, where $w_i^{(v)}$ denotes the weight assigned to the $v$-th view at temporal position $i$.

**Feature Fusion via Weighted Aggregation.** Once the view weights are computed for each temporal position, the final fused features are obtained through weighted aggregation. The fused feature at each temporal position is computed as a weighted combination of the view-specific features:

$$\mathbf{h}_{\text{fused},i} = \sum_{v=1}^{V} w_i^{(v)} \cdot \mathbf{h}_{v,i}$$

where $\mathbf{h}_{v,i}$ denotes the feature vector from view $v$ at temporal position $i$, and the sum is taken over all three views. This weighted aggregation operation ensures that each temporal position can emphasize the views that are most informative at that particular time instant. The temporal flexibility of this approach is crucial for industrial fault diagnosis, as different frequency components and temporal characteristics may be more salient at different time periods during the evolution of a fault.

The aggregated features are then processed through an output projection layer that applies a linear transformation to maintain the original feature dimensionality:

$$\mathbf{h}_{\text{fused}}^{\text{final}} = \text{Dropout}(\mathbf{h}_{\text{fused}} \mathbf{W}_O + \mathbf{b}_O)$$

where $\mathbf{W}_O \in \mathbb{R}^{C \times C}$ and $\mathbf{b}_O \in \mathbb{R}^{C}$ are the projection parameters, and dropout regularization with a rate of $0.1$ is applied to prevent overfitting. The resulting fused representation $\mathbf{h}_{\text{fused}}^{\text{final}} \in \mathbb{R}^{C \times T}$ constitutes a comprehensive feature representation that integrates information from all three domain perspectives through learned, adaptive importance weights.

### 3.1.3 Advantages of the Tiny Attention Fusion Approach

The tiny attention mechanism offers several advantages over alternative fusion strategies. First, the mechanism learns view-specific weights directly from the training data, allowing the model to adaptively discover which views are most important given the input signal characteristics. This data-driven approach is more flexible than fixed fusion rules such as simple concatenation or equal weighting, which cannot adjust to varying signal properties.

Second, the computational efficiency of the tiny attention mechanism is significantly higher compared to more complex fusion approaches. The weight network consists only of two fully connected layers with a dimensionality reduction step, resulting in a linear computational cost with respect to the sequence length. This is in stark contrast to sequence-level attention mechanisms that operate on the temporal dimension and incur quadratic complexity in the sequence length.

Third, the per-timestep weighting enables the mechanism to assign different importance levels to different views at different time instants. This temporal flexibility is particularly valuable for fault diagnosis applications, where the relevance of different frequency components and temporal patterns may vary during the fault evolution. The mechanism can emphasize the time-domain view during periods of impulsive transients while emphasizing frequency-domain information during stationary fault phases.

Finally, the learned weights provide interpretability into the model's fusion decisions. The scalar weights $w_i^{(v)}$ directly indicate which views the model deems important at each temporal location, offering insights into the signal characteristics and the model's reasoning process. This interpretability is important for industrial applications where understanding and validating model decisions is essential.

### 3.1.4 Mathematical Formulation Summary

The complete multi-view feature encoding and fusion process can be formulated as a functional composition. Let the multi-view feature encoding be denoted as:

$$\mathcal{H} = \{\mathcal{F}_{\text{time}}(\mathbf{x}), \mathcal{F}_{\text{freq}}(\mathbf{x}), \mathcal{F}_{\text{wavelet}}(\mathbf{x})\}$$

The tiny attention-based fusion operation is then expressed as:

$$\mathbf{h}_{\text{fused}}^{\text{final}} = \mathcal{G}_{\text{tiny}}(\mathcal{H}; \Theta_{\text{attention}})$$

where $\mathcal{G}_{\text{tiny}}(\cdot)$ represents the tiny attention fusion operator and $\Theta_{\text{attention}} = \{\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2, \mathbf{W}_O, \mathbf{b}_O\}$ denotes the complete set of learnable parameters in the fusion mechanism.

The core operation of the tiny attention mechanism can be expressed as a composition of the weight computation and weighted aggregation operations:

$$\mathbf{w} = \text{Softmax}(\text{ReLU}(\mathcal{H} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2)$$

$$\mathbf{h}_{\text{fused}} = \sum_{v=1}^{V} \mathbf{w}^{(v)} \odot \mathbf{h}_v$$

where $\odot$ denotes element-wise multiplication applied across the feature dimensions. The learned weights $\mathbf{w}$ provide a mechanism for the model to dynamically adjust the contribution of each view based on the input characteristics, while the weighted aggregation ensures that the final representation combines information from all three views in a principled manner.

The fused representation $\mathbf{h}_{\text{fused}}^{\text{final}}$ serves as the comprehensive multi-view feature input to subsequent processing modules, such as the latent diffusion model for feature synthesis. By integrating heterogeneous representations through learned, temporal-aware weights, the tiny attention mechanism enables the model to capture the full richness of information distributed across different signal domains while maintaining computational efficiency and interpretability.
