---
layout: wiki
title: 2023-05-04：论文速递
cate1: paper
cate2:
description: 
keywords: paper
mathjax: true
---


## 2023-05-04：论文速递

### Unlimiformer: Long-Range Transformers with Unlimited Length Input

> 

> 《Unlimiformer: Long-Range Transformers with Unlimited Length Input》
>
> - URL：https://arxiv.org/abs/2305.01625
>
> - Official Code：https://github.com/abertsch72/unlimiformer
>
> - 单位：Carnegie Mellon University

- 动机：现有的 transformer 模型由于需要考虑每 **输入 Token**，因此通常对其 **输入长度有限制**，但实际应用中需要能够处理任意长度的输入。

- 方法：提出一种名为 Unlimiformer 的方法，可以扩展现有的预训练编-解码器 Transformer 模型，使其在测试时能够接受任意长度的输入，而 **无需修改模型代码或添加学习的权重**。Unlimiformer 通过构建一个**数据存储库（datastore）**，将所有输入符号的隐藏状态储存起来，并使用一种新的 **k-近邻索引技术**，将注意力计算分散在所有层上。

- 优势：Unlimiformer 可以应用于多个基础模型，例如 BART 或 PRIMERA，而不需要添加权重或重新训练。在多个长文档和多文档摘要基准测试上，Unlimiformer 的性能优于其他强大的长程 Transformer 模型，并且随着输入 Token 长度的增加，Unlimiformer 的推理时间呈 **亚线性（sublinearly）增长**。

> 相比于现有的接收长输入 Token 序列的 Transformer（例如最近的 Transformer 模型能够处理 200W 的 Token 序列），Unlimiformer 这种方式在工程上更加容易实施，并且可以利用现有的、可以利用的 Transformer 模型，而不需修改 Transfomrer 模型代码或者重新训练一个 Transformer 模型。


#### 前言

现有的大型数据集，如下图所示：

![现有大型数据集的平均 Token 长度](https://pic2.zhimg.com/80/v2-b0b83bfe2df760b436996a95886572e9_720w.webp)



#### Unlimiformer 框架

Unlimiformer 的框架图如下所示：

![Unlimiformer 框架图](https://pic4.zhimg.com/80/v2-9a791d17c1e39f459bf9505c4b9f1c87_720w.webp)

图中已输入 6 个 Token 序列为例，假设 Transformer 模型最多只能处理 2 个 Token 的序列。



![](https://pic4.zhimg.com/80/v2-e48ebd601ddcccc762818c00dcb8919f_720w.webp)

随着最大数据存储大小的增加，实体召回通常会增加。在所有数据存储大小下，Unlimiformer 都优于 BART 基线（红色虚线）。




## 参考

- 知乎：[爱可可AI前沿推介(5.4)](https://zhuanlan.zhihu.com/p/626548701)

