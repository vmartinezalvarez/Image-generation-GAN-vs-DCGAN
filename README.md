# MNIST Image Generation:  GAN vs DCGAN

TensorFlow 2.0 implementation of Generative Adversarial Networks (GAN) [1] and Deep Convolutional Generative Adversarial Networks (DCGAN) [2] for MNIST [3] dataset.

## Results

<table align='center'>
<tr align='center'>
<td> GAN</td>
<td> DCGAN</td>
</tr>
<tr>
<td><img src = 'MNIST_GAN_results/generation_animation.gif'>
<td><img src = 'MNIST_DCGAN_results/MNIST_DCGAN_generation_animation.gif'>
</tr>
</table>

* MNIST vs Generated images

<table align='center'>
<tr align='center'>
<td> MNIST </td>
<td> GAN after 100 epochs </td>
<td> DCGAN agter 20 epochs </td>
</tr>
<tr>
<td><img src = 'MNIST_GAN_results/raw_MNIST.png'>
<td><img src = 'MNIST_GAN_results/MNIST_GAN_100.png'>
<td><img src = 'MNIST_DCGAN_results/MNIST_DCGAN_20.png'>
</tr>
</table>

* Training loss
  * GAN


* Learning time
    * MNIST GAN - CPU times: user 12min 58s, sys: 2min 21s, total: 15min 19s. Wall time: 15min 8s
    * MNIST DCGAN - Avg. per epoch: 175.84 sec; Total 20 epochs: 3619.97 sec


## Reference

[1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.

(Paper: http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

[2] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

(Paper: https://arxiv.org/pdf/1511.06434.pdf)

[3] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, Nov 1998.
