---
layout: post
author: bookstall
tags: SSM
categories: [SSM]
excerpt: 第一篇关于 Mamba 在计算机视觉领域中应用的综述
keywords: SSM
title: The Survey of Vision Mamba
mathjax: true
---

全面综述 Visual Mamba 的发展，包括基本概念、用于视觉任务的适配设计、与其他模块的集成，以及在不同视觉任务中的应用，揭示这一新兴架构在计算机视觉领域的巨大潜力。

## 前言

- 全面综述了视觉 Mamba，即被适配用于计算机视觉任务的状态空间模型（SSM）。

- 介绍了 SSM 的关键概念，包括状态空间公式，离散化，GPU内存利用，以及让 Mamba 比传统 SSM 更强大的选择机制。

- 对于视觉任务，回顾了适配 Mamba 模块（如 ViM、VSS）和扫描机制以处理图像和视频等多维数据的工作。

- 将方法分类为纯 Mamba 模型和将 Mamba 与卷积、循环、注意力等其他技术相结合的模型。

- Mamba 在高级视觉(检测、分割)、低级视觉(超分辨率、生成)、医学图像任务上都展现出了非常有前景的结果。

- 在提高效率、降低计算复杂度、增强与其他架构的集成等方面仍存在挑战。


## ViM 和 VSS Block


![ViM 和 VSS 两种模块的示意图](/images/posts/Survey-Mamba-Vision/ViM-and-VSS-block.png)



## 选择性扫描机制


![各种扫描机制的示意图](/images/posts/Survey-Mamba-Vision/selective-scan-methods.png)



![扫描机制的方法总结](/images/posts/Survey-Mamba-Vision/selective-scan-tables.png)

## 参考

- [A Survey on Visual Mamba](https://arxiv.org/abs/2404.15956)





