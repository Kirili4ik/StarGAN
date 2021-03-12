# StarGAN model in PyTorch

This repository is an implementation of model described in [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/pdf/1711.09020v3.pdf). 

Model was trained on CelebA dataset. It can be found in Torchvision (or via link in main.ipynb).

Main file with is train/test loops in a notebook(main.ipynb). It also contains config so that all hyperparameters can be found there.

WandB loffing and report can be found on their web-page ([report](https://wandb.ai/kirili4ik/dgm-ht2/reports/-2-DGM-StarGan---Vmlldzo1MjE1NTk), [logging](https://wandb.ai/kirili4ik/dgm-ht2?workspace=user-kirili4ik)) (in Russian).

There are some good examples of trained models:
![alt text](https://github.com/Kirili4ik/StarGAN/blob/main/samples_base.png "samples")

Also there were some discoveries and experiements, e.g. FID architectures comparison:
![image](https://user-images.githubusercontent.com/30757466/110873817-a279a400-82e3-11eb-890c-e1bab8991086.png)

As a result of experiements IN are replaced with BN in Generator for now. Also ConvTranspose layers are replaced with Upsample+Conv layers as described [here](https://distill.pub/2016/deconv-checkerboard/). Training was performed on Nvidia 2080 TI with pics resized to 128x128 and batch size of 32.

Thx [mseitzer](https://github.com/mseitzer/pytorch-fid) for implementing Inception.py for FID calculation.

Thx [WanbB](https://wandb.ai/) for convenient logging and beautiful report writing tool.
