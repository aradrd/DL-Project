The dataset contains MRI samples of 2 forms: [[Multi-Coil]] and single-coil (our version only has the single-coil variant).
The single-coil is generated from the multi-coil.
Aside from the slices, which are 2d representations of the 3d MRI scan, the dataset also contains 2 types of "reconstructions" (which are actually just a transformed non sub-sampled kspace data in the image domain): 
1. [[Multi-Coil|rss]] from the multi-coil data (which we don't have).
2. ESC (inverse Fourier transform) from the single-coil data
