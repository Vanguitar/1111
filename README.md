## Multi-view Feature Encoding and Fusion with Tiny Attention Mechanism

Industrial vibration signals contain rich information distributed across multiple domains. To obtain comprehensive representations of fault characteristics, the proposed framework extracts features from three complementary domains and fuses them through a view-level attention mechanism. This section describes the multi-view feature extraction and fusion strategy.

### 3.1.1 Three-View Feature Extraction

Given an input signal $\mathbf{x} \in \mathbb{R}^{L}$ of length $L$, the framework computes feature representations from three distinct domains:

**Time-Domain View.** The time-domain features are extracted by processing the raw vibration signal through a one-dimensional convolutional encoder:

$$\mathbf{h}_t = \mathcal{F}_{\text{time}}(\mathbf{x}; \Theta_t)$$

where $\mathcal{F}_{\text{time}}(\cdot; \Theta_t)$ denotes the time-domain CNN encoder with learnable parameters $\Theta_t$, and $\mathbf{h}_t \in \mathbb{R}^{C \times T}$ represents the extracted features with channel dimension $C$ and temporal sequence length $T$.

**Frequency-Domain View.** The frequency spectrum is obtained by applying the Fast Fourier Transform to the input signal:

$$\mathbf{X}_{f} = \text{FFT}(\mathbf{x})$$

The magnitude spectrum is extracted as $\tilde{\mathbf{X}}_f = |\mathbf{X}_f| \in \mathbb{R}^{L/2+1}$. This magnitude spectrum is then processed through an identical one-dimensional convolutional architecture:

$$\mathbf{h}_f = \mathcal{F}_{\text{freq}}(\tilde{\mathbf{X}}_f; \Theta_f)$$

where $\mathbf{h}_f \in \mathbb{R}^{C \times T}$ denotes the frequency-domain features and $\Theta_f$ contains the encoder parameters.

**Time-Frequency-Domain View.** The continuous wavelet transform is employed to capture multi-scale temporal-spectral information:

$$\mathbf{W}_{s,\tau} = \int_{-\infty}^{+\infty} \mathbf{x}(t) \psi_{s,\tau}^{*}(t) dt$$

where $\psi_{s,\tau}(t) = \frac{1}{\sqrt{s}}\psi\left(\frac{t-\tau}{s}\right)$ is the scaled and shifted wavelet function, with scale parameter $s$ and time shift $\tau$. The magnitude coefficients $|\mathbf{W}_{s,\tau}|$ are arranged into a two-dimensional representation and resized to a standard dimension of $64 \times 64$:

$$\mathbf{M}_{w} = \text{Resize}(|\mathbf{W}_{s,\tau}|; 64 \times 64)$$

A two-dimensional convolutional encoder then processes this representation:

$$\mathbf{h}_w = \mathcal{F}_{\text{wavelet}}(\mathbf{M}_{w}; \Theta_w)$$

yielding the time-frequency-domain features $\mathbf{h}_w \in \mathbb{R}^{C \times T}$, where $\Theta_w$ denotes the encoder parameters.

### 3.1.2 View-Level Attention Fusion

After extracting heterogeneous representations from the three views, a view-level attention mechanism is employed to compute adaptive weights for each view and fuse them into a comprehensive feature representation. This approach differs from sequence-level attention mechanisms by operating at the view level, enabling the model to learn the importance of each entire view domain.

**View Feature Stacking.** The three view feature tensors are first stacked together:

$$\mathcal{V} = \text{Stack}([\mathbf{h}_t, \mathbf{h}_f, \mathbf{h}_w]) \in \mathbb{R}^{B \times V \times C \times T}$$

where $B$ denotes the batch size, $V=3$ is the number of views, and the result has shape $(B, 3, C, T)$. The stacked tensor is then reshaped for per-timestep processing:

$$\mathcal{V}_{\text{reshaped}} = \text{Reshape}(\mathcal{V}; (B \cdot T, V, C)) \in \mathbb{R}^{B \cdot T \times V \times C}$$

This reshaping operation rearranges the tensor dimensions to process each temporal position across all views independently, effectively treating the fusion as a per-position operation.

**View Weight Computation.** For each temporal position, the view-level attention mechanism computes a weight for each view through a lightweight neural network. Let $\mathcal{V}_{reshaped}^{(i)} \in \mathbb{R}^{V \times C}$ denote the features at temporal position $i$. The view weights are computed as:

$$\mathbf{w}_i = \text{ViewWeightNet}(\mathcal{V}_{\text{reshaped}}^{(i)}; \Theta_{\text{weight}})$$

where the ViewWeightNet is a small multi-layer perceptron defined as:

$$\mathbf{a}_i = \text{ReLU}(\mathcal{V}_{\text{reshaped}}^{(i)} \mathbf{W}_1 + \mathbf{b}_1)$$

$$\mathbf{w}_i = \text{Softmax}(\mathbf{a}_i \mathbf{W}_2 + \mathbf{b}_2)$$

where $\mathbf{W}_1 \in \mathbb{R}^{C \times (C/4)}$ is the first weight matrix that projects from the channel dimension $C$ to $C/4$, and $\mathbf{W}_2 \in \mathbb{R}^{(C/4) \times V}$ projects to $V$ view dimensions. The softmax operation ensures that $\mathbf{w}_i \in \mathbb{R}^{V}$ sums to one, yielding a valid probability distribution over the views.

**Weighted View Fusion.** Once the view weights are computed, the final fused feature is obtained through weighted summation:

$$\mathbf{h}_{\text{fused},i} = \sum_{v=1}^{V} w_i^{(v)} \cdot \mathcal{V}_{\text{reshaped},i}^{(v)}$$

where $w_i^{(v)}$ is the $v$-th weight for temporal position $i$, and $\mathcal{V}_{\text{reshaped},i}^{(v)}$ is the feature vector of the $v$-th view at position $i$. This element-wise weighted summation produces fused features $\mathbf{h}_{\text{fused},i} \in \mathbb{R}^{C}$ for each temporal position.

**Output Projection and Reshaping.** After fusion, an output projection layer and dropout are applied for regularization and dimensionality adjustment:

$$\mathbf{h}_{\text{projected},i} = \text{Dropout}(\mathbf{h}_{\text{fused},i} \mathbf{W}_O + \mathbf{b}_O)$$

where $\mathbf{W}_O \in \mathbb{R}^{C \times C}$ and $\mathbf{b}_O \in \mathbb{R}^{C}$ are the output projection parameters. Finally, the features are reshaped back to the original tensor format:

$$\mathbf{h}_{\text{fused}} = \text{Reshape}([\mathbf{h}_{\text{projected},1}, \ldots, \mathbf{h}_{\text{projected},B \cdot T}]; (B, C, T))$$

The resulting tensor $\mathbf{h}_{\text{fused}} \in \mathbb{R}^{B \times C \times T}$ is the final fused representation that integrates information from all three views in a learned, adaptive manner.

### 3.1.3 Advantages of View-Level Attention Fusion

The view-level attention fusion mechanism offers several advantages over alternative fusion strategies. First, by learning view-specific weights, the mechanism can adaptively adjust the contribution of each domain based on the input signal characteristics. Unlike fixed fusion rules (e.g., simple concatenation or equal weighting), the view-level attention learns data-driven importance weights directly from the training data.

Second, the lightweight nature of the weight network ensures computational efficiency. Rather than computing complex pairwise interactions between views through multi-head cross-attention, the view-level attention requires only a simple multi-layer perceptron to compute scalar weights for each view. This design choice reduces computational overhead while maintaining the capacity to capture view-specific information relevance.

Third, the per-timestep weighting enables the mechanism to assign different importance to different views at different temporal locations. This temporal flexibility allows the model to emphasize views that are more informative at specific time instants, which is particularly valuable for transient fault detection where different frequency components may be salient at different times.

Finally, the interpretability of view weights is enhanced compared to complex attention mechanisms. The learned weights $\mathbf{w}_i$ directly indicate which views are deemed important at each temporal position, providing insight into the model's fusion decisions.

### 3.1.4 Mathematical Formulation Summary

The complete view-level fusion operation can be formulated as a composition of functions. Let $\mathcal{G}_{\text{view}}$ denote the view fusion operator:

$$\mathbf{h}_{\text{fused}} = \mathcal{G}_{\text{view}}(\{\mathbf{h}_t, \mathbf{h}_f, \mathbf{h}_w\}; \Theta_{\text{view}})$$

where $\Theta_{\text{view}} = \{\mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_O, \mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_O\}$ denotes the complete set of learnable parameters in the view fusion module. The fusion operation can be decomposed as:

$$\mathcal{G}_{\text{view}} = \text{Reshape} \circ \text{Dropout} \circ \text{Proj} \circ \text{WeightedSum} \circ \text{WeightNet} \circ \text{Reshape}$$

where each composition represents a sequential operation in the pipeline.

The learned view weights provide interpretable information about domain importance. For a given temporal position $i$ and view $v$, the weight $w_i^{(v)}$ can be analyzed to understand which domains contribute most to the final representation at different temporal locations. This interpretability is important for understanding the model's behavior in industrial fault diagnosis applications where domain knowledge is valuable.

The fused representation $\mathbf{h}_{\text{fused}}$ serves as the comprehensive feature input to subsequent modules (e.g., the latent diffusion model for feature synthesis), ensuring that unseen-domain features are generated from a balanced and adaptively-weighted combination of time-domain, frequency-domain, and time-frequency-domain information.
