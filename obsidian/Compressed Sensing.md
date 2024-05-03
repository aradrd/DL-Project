Used to reduce amount of frequencies sampled in MRI session (reduce time needed).
1. Transformation to the [[Frequency Domain]].
2. Sub-sampling (e.g. arbitrary values are zeroed-out), still in the [[Frequency Domain]].
	1. Denote the sub-sampled binary mask $\mathcal M$ and the sub-sampled dataset $\mathcal M (\tilde x) = \tilde x _ \text{subsampled}$.
3. Transformation back to the [[Image Domain]].
	1. Denote the transformed sub-sampled batch as $x_ \text{subsampled}$.
4. Feed into a [[Neural Network]] for reconstruction.
	1. Denoted $x_{\text{reconstructed}} = m_\theta(x_\text{subsampled})$.
Goal:
$$m_\theta\left(\mathcal M(\tilde x)\right) \approx x$$
![[Compressed Sensing.png|500]]
## Levels of subsampling
- Random subsampling mask $\mathcal M$.
- Parametric subsampling mask $\mathcal M _ \psi$.