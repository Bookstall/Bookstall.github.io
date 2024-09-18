---
layout: post
author: bookstall
tags: Transformer, RoPE
categories: [Transformer, RoPE]
excerpt: DeepSeek-V2
keywords: Transformer, RoPE
title: RoPE-Tie（RoPE for Text-image）
mathjax: true
---

本文直接引用自 [Transformer升级之路：17、多模态位置编码的简单思考](https://spaces.ac.cn/archives/10040) 里面的内容，仅供自用，侵删。

## 前言

在这个系列的第二篇文章[《Transformer升级之路：2、博采众长的旋转式位置编码》](https://spaces.ac.cn/archives/8265)中，笔者提出了旋转位置编码（RoPE）——通过绝对位置的形式实现相对位置编码的方案。

一开始 RoPE 是针对一维序列如文本、音频等设计的（**RoPE-1D**），后来在[《Transformer升级之路：4、二维位置的旋转式位置编码》](https://spaces.ac.cn/archives/8397)中我们将它推广到了二维序列（**RoPE-2D**），这适用于图像的 ViT。

然而，不管是 RoPE-1D 还是 RoPE-2D，它们的共同特点都是 **单一模态**，即 **纯文本或者纯图像输入** 场景，那么对于多模态如图文混合输入场景，RoPE 该做如何调整呢？

笔者搜了一下，发现鲜有工作讨论这个问题，主流的做法似乎都是直接展平所有输入，然后当作一维输入来应用 RoPE-1D，因此连 RoPE-2D 都很少见。且不说这种做法会不会成为图像分辨率进一步提高时的效果瓶颈，它终究是显得不够优雅。所以，接下来我们试图探寻两者的一个自然结合。





## 参考

- 科学空间：[Transformer升级之路：17、多模态位置编码的简单思考](https://spaces.ac.cn/archives/10040)

