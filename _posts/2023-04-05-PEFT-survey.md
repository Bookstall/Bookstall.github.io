---
layout: post
author: bookstall
tags: PEFT
categories: [PEFT]
excerpt: PEFT（Parameter-Efficient Fint-Tuning）的综述
keywords: PEFT
title: 综述：PEFT（Parameter-Efficient Fint-Tuning）
mathjax: true
sticky: False
---

本文是对最新的（2023 年 4 月）综述论文 [《Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning》](http://arxiv.org/abs/2303.15647) 的一个阅读记录。

论文作者介绍了三大类主要的 PEFT 方法：

- 基于加法（Additive）

- 基于选择（Selective）

- 基于重新参数化（Reparametrization-based）。

在基于加法的方法中，又额外区分了两大类：

- 类适配器方法（Adapters）

- 软提示（Soft Prompts）

![Serval PEFT methods](/images/posts/PEFT-Survey/peft-methods-taxonomy.png)

![Serval SOTA PEFT methods in Transformer](/images/posts/PEFT-Survey/several-sota-peft-methods-in-transformer.png)

![PEFT 方法的评估结果](/images/posts/PEFT-Survey/PEFT-evaluated.png)

![PEFT 方法在存储和内存方面的比较](/images/posts/PEFT-Survey/PEFT-storage-memory-computational.png)

![PFET 方法的分类图](/images/posts/PEFT-Survey/PEFT-Survey.svg)

## 1、Additive




## 2、Selective




## 3、Reparameterization-based（重参数化）





## 4、Hyperactive（混合、组合）




## 参考

- 综述论文：[《Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning》](http://arxiv.org/abs/2303.15647)

- 苏剑林博客：[Ladder Side-Tuning：预训练模型的“过墙梯”](https://kexue.fm/archives/9138)

- LoRA：[《LoRA: Low-Rank Adaptation of Large Language Models》](https://arxiv.org/abs/2106.09685)

- MIM Adapter：
  
  - 知乎：[ICLR2022 高分文章：将 Adapter、prompt-tuning、LoRA 等纳入统一的框架](https://zhuanlan.zhihu.com/p/436571527)



