# MNIST Image Generation:  GAN vs DCGAN

TensorFlow 2.0 implementation of Generative Adversarial Networks (GAN) [1] and Deep Convolutional Generative Adversarial Networks (DCGAN) [2] for MNIST [3] dataset.

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
<td> DCGAN agter 20 epochs </td>
</tr>
<tr>
<td><img src = 'MNIST_GAN_images/image_at_epoch_0200.png'>
<td><img src = 'MNIST_DCGAN_images/image_at_epoch_0100.png'>
</tr>
</table>


* Learning time
    * MNIST GAN - CPU times: user 12min 58s, sys: 2min 21s, total: 15min 19s. Wall time: 15min 8s
    * MNIST DCGAN - Avg. time for epoch is 26.319965600967407 sec


## Reference

[1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.

(Paper: http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

[2] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

(Paper: https://arxiv.org/pdf/1511.06434.pdf)

[3] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, Nov 1998.
