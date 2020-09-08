# MNIST Image Generation:  GAN vs DCGAN

TensorFlow 2.0 implementation of Generative Adversarial Networks (GAN) [1] and Deep Convolutional Generative Adversarial Networks (DCGAN) [2] for MNIST [3] dataset.

## Abstract

Recently, the search for creative artificial intelligence has turned to generative adversarial network (GAN), which is currently one of the most popular and successful application of deep learning. Motivated by the ability of GANs to sampling
from a latent space of images to create entirely new images, here we evaluate and compare the
performance of a GAN, where both generator and discriminator are multilayer perceptrons with a
deep convolutional generative adversarial network (DCGAN) on the MNIST dataset.

## Results

<table align='center'>
<tr align='center'>
<td> GAN</td>
<td> DCGAN</td>
</tr>
<tr>
<td><img src = 'MNIST_GAN_images/gan.gif.png'>
<td><img src = 'MNIST_DCGAN_images/dcgan.gif.png'>
</tr>
</table>

* GAN vs DCGAN

<table align='center'>
<tr align='center'>
<td> GAN after 200 epochs </td>
<td> DCGAN after 100 epochs </td>
</tr>
<tr>
<td><img src = 'MNIST_GAN_images/image_at_epoch_0200.png'>
<td><img src = 'MNIST_DCGAN_images/image_at_epoch_0100.png'>
</tr>
</table>


* Learning time
    * MNIST GAN - Avg. time for epoch  is 4.564566612243652 sec
    * MNIST DCGAN - Avg. time for epoch is 26.319965600967407 sec


## Reference

[1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.

(Paper: http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

[2] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

(Paper: https://arxiv.org/pdf/1511.06434.pdf)

[3] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, Nov 1998.
