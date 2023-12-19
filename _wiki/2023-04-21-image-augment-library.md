---
layout: wiki
title: 图像数据增强库：albumentations 和 kornia.augmentation
cate1: PyTorch
cate2:
description: 
keywords: PyTorch
mathjax: true
---

## 图像数据增强库

`albumentations` 和 `kornia.augmentation` 都是图像数据增强库，并且都支持 PyTorch，主要的不同之处在于：

- `albumentations` 仅支持使用 **CPU** 处理图像数据增强

- `kornia` 支持使用 **GPU** 处理图像数据增强，并且支持同时处理多个图像（即 batch_size > 1）

  - 相比于 NVIDIA 的 [DALI](https://github.com/NVIDIA/DALI)，更加容易使用

---

`kornia` 本身是一个可微的计算机视觉库，`kornia.augmentation` 只是 `kornia` 的其中一个模块。即，`kornia.augmentation` is a module to perform data augmentation in the GPU.

### 1、albumentations

> **Fast image augmentation library** and an easy-to-use wrapper around other libraries.
>
> `albumentations` 是一个用于图像增强的 Python 库。图像增强用于深度学习和计算机视觉任务，以提高训练模型的质量。图像增强的目的是从现有数据创建新的训练样本。

- 文档：https://albumentations.ai/docs/

- 论文：[Albumentations: Fast and Flexible Image Augmentations](https://www.mdpi.com/2078-2489/11/2/125)

安装如下所示：

```shell
pip install -U albumentations
```

下面的图像数据增强的例子：

![albumentations 图像数据增强的例子](https://camo.githubusercontent.com/3bb6e4bb500d96ad7bb4e4047af22a63ddf3242a894adf55ebffd3e184e4d113/68747470733a2f2f686162726173746f726167652e6f72672f776562742f62642f6e652f72762f62646e6572763563746b75646d73617a6e687734637273646669772e6a706567)


### 2、kornia

> - Kornia is a differentiable computer vision library for PyTorch.
>
> -     Kornia 是 PyTorch 的可微分计算机视觉库。
>
> - It consists of a set of routines and differentiable modules to solve generic computer vision problems. At its core, the package uses PyTorch as its main backend both for efficiency and to take advantage of the reverse-mode auto-differentiation to define and compute the gradient of complex functions.
> 
>   - 它由一组例程和可微模块组成，用于解决一般的计算机视觉问题。该包的核心是使用 PyTorch 作为其主要后端，以提高效率并利用反向模式自动微分来定义和计算复杂函数的梯度。

- 论文：[Kornia: an Open Source Differentiable Computer Vision Library for PyTorch](https://arxiv.org/abs/1910.02190)

- 会议：WACV 2020

- 入门 Tutorial：https://kornia-tutorials.readthedocs.io/en/latest/

- 图像数据增强的文档：https://kornia.readthedocs.io/en/latest/augmentation.html

安装如下所示：

```shell
pip install kornia

# 或者
pip install git+https://github.com/kornia/kornia
```

`kornia` 还提供了一个关于 `kornia.augmentation` 的 [HuggingFace Demo](https://huggingface.co/spaces/kornia/kornia-augmentations-tester)，可以自行编写相应的图像变换的代码。


## 参考

- github repo：[albumentations](https://github.com/albumentations-team/albumentations)

- github repo：[kornia](https://github.com/kornia/kornia)



