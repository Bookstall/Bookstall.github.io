---
layout: wiki
title: 2023-05-15：论文速递
cate1: paper
cate2:
description: 
keywords: paper
mathjax: true
---

## 2023-05-15：论文速递

### Cuttlefish: Low-Rank Model Training without All the Tuning

> Cuttlefish：无需微调的低秩模型训练

> 《Cuttlefish: Low-Rank Model Training without All the Tuning》
>
> - URL：https://arxiv.org/abs/2305.02538
>
> - Official Code：https://github.com/hwang595/Cuttlefish
>
> - 会议：MLSys 2023
>
> - 单位：CMU & University of Wisconsin-Madison & Sony Group Corporation
>
> 

- 动机：低秩神经网络训练可以有效地减少可训练参数的数量，加速端到端的速度，但需要调整多个额外的因子分解超参数，本文旨在解决这一挑战。

- 方法：提出一种自动化的低秩训练方法 Cuttlefish，通过观察神经网络层稳定秩的变化，自适应选择每层的秩和完全秩热身训练时间，不需要调整因子分解超参数。

- 优势：Cuttlefish 自动化地选择因子分解超参数，不需要多次试验进行调整，生成的模型比完全秩模型小 5.6 倍，同时保持可比较的准确性，比其他低秩模型训练方法和基线方法表现更好。

<a href="https://pic2.zhimg.com/80/v2-15fe119a1283205459f0fdf39d0d2b99_720w.webp" data-fancybox="images"><img src="https://pic2.zhimg.com/80/v2-15fe119a1283205459f0fdf39d0d2b99_720w.webp" alt="" style="
    zoom: 67%;
"></a>

![](https://pic1.zhimg.com/80/v2-c4323f0657d5eb99402a3184bdfa1f4c_720w.webp)

![](https://pic3.zhimg.com/80/v2-835f0b0494e26be7d0577c472bc59a56_720w.webp)


## 参考

- 知乎：[爱可可AI前沿推介(5.15)](https://zhuanlan.zhihu.com/p/629373247)

