# Spine_project
Spine discs segmentation


The dataset is provided by MICCAI challenge 2018.
The dataset contains 16 3D MRI images. Every MRI Image consists of 4 modalities of 'wat', 'fat', ‘inn', and ‘opp'.
Here I implemented a U-Net to segment the disc area in spine images. 

inputs: 2d images i.g., slices of 3d images
output: 2d masks segmenting disc area 

This code performed with 91% accuracy in 6 testing 3D images.  
