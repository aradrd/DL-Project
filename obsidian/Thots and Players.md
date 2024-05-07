Using generative models our noising process will be the subsampling (whose mask can also be learned), and our denoising process will hopefully reconstruct the original image as best as possible (somehow subsampling can be undone by small [[Gaussian]] denoising steps).
Models that we've seen in the course and might be effective for our task:
- CNN (vanilla)
- VAE
- Diffusion
	- UNet architecture seems to be the way to go (for any task?), there are many versions and optimizations we should look into.
We can gather more models from:
- Papers
	- PILOT
		- More focused on physical constraints and less on modern architecture, optimizer, loss function (which are the exact things we should focus on...) but a good starting point nonetheless.
		- Links to many papers that might also be relevant and discuss reconstruction methods and models.
		- Mentions that dropping higher frequencies improves results. This could be included by implementing penalties for not choosing high frequencies as some form of normalization.
- Ofek
	- [ ] Set meeting
- Doda (maybe)
