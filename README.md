"Image Captioning" is a TensorFlow based implementation of a system that receives images as input and produces 
descriptions of these images using Deep Learning techniques.

It is based on the paper of Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan, "Show and Tell: A Neural Image Caption Generator" (https://arxiv.org/abs/1411.4555).

More specifically, we used a CNN to exctract features from the images connected to an LSTM that was responsible for the 
captions generation.
Moreover, we used an image embedder in order to represent our images, whereas our words were represented by Word Embedding Vectors.
The model was trained in an end-to-end fashion on the MSCOCO 2015 dataset.
