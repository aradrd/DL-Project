Working with [[Generative Models]] has 2 parts:
1. Training
	1. Noising Process: We apply [[Gaussian]] noise to the original image.
	2. Denoising Process: One or more steps are used to remove the noise assuming [[Gaussian]] distribution with learned parameters.
2. Inference: This step uses the trained model for denoising to remove noise from some assumed unseen input.
	- This can also be used for generation tasks where the input is sampled from a learned (static?) distribution space used in the noising process.
![[Generative Models.jpg|650]]