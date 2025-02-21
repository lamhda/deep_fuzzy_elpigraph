## Deep Fuzzy ElPiGraph method

The method uses a combination of variational graph autoencoders with the base algorithm of elastic principal graphs, [ElPiGraph](https://github.com/j-bac/elpigraph-python). The general idea is to organize the latent spaces of complex autoencoders as a set of branching trajectories such that the latent spaces could be more efficiently explored for the generative data modeling. 


Figure exemplifying the approach.


The method suggest the core module implemented as a standard torch.nn.Module Pytorch module that can be combined with various types of encoders and decoders, including simple autoencoders, VQ-VAE, but also with convolutional and graph encoders that make the application of the method suitable for the analysis of non-tabular data (images, graphs, chemical structures, texts) with the aim of trajectory discovery in the latent spaces of such datasets. 

Currently the module is tested with simple autoencoders and synthetic generation of datasets possessing the geometrical structure with branching trajectories. 
