# xt-vae: 
## a simple variational autoencoder for single cell transcriptomics data

This was an experiment with using `tensorflow-probability` package to build a variational autoencoder.  It's tightly coupled to this [single-cell RNA data](https://celltypes.brain-map.org/rnaseq/mouse_ctx-hip_10x) from the Allen Institute.  

For fun I decided to use a 3D latent space, so there is a utility for saving the output a PLY file full of spheres with vertex colors.  You can open it up in Blender, wire up the material to be driven by vertex colors, and make something like this:



Blender looks like this when you do:

<img src="https://github.com/dyf/xt-vae/blob/main/blender_screenshot.png?raw=true">
