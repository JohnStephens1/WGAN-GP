My personal implementation of a WGAN-GP (Wasserstein Generative Adverserial Neural Network with Gradient Penalty) with the goal of producing paintings in the style of Monet and generating images of cats.

The use of the Wasserstein loss adresses a few issues regarding traditional GANs, resulting in improved stability during training and higher quality of generated images.

Performance of GANs tends to be significantly better when the discriminator obeys a Lipschitz constraint, essentially limiting the maximum gradient. Traditionally, this is done using clipping, resulting in obvious shortcomings. In my implementation, following the linked paper<sup>[1]</sup>, I have implemented Gradient Penalty, a softer constraint enforced by a penalty on the gradient norm for random samples.

The final quality of generated images could be significantly improved by adding layers and increasing their complexity. Due to significant hardware limitations on my behalf, this is the most my radiator is willing to give.

[1]. [The paper](https://arxiv.org/pdf/1704.00028.pdf)
