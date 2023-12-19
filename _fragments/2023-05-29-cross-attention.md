---
layout: fragment
title: Cross-Attention（交叉注意力）
tags: [Transformer]
excerpt: Cross-Attention（交叉注意力）
keywords: Transformer
mathjax: true
---


## Cross-Attention

在多模态学习领域中，**交叉注意力（Cross-Attention）** 是一种常用的模态融合方式。

假设现在有 Video 和 Text 两种模态的数据，如果要将 Text 的数据融合到 Video 中，只需要将 Attention 中的 `Query = Video`，`Key & Value = Text`，然后进行计算即可：

```python
# Post-Norm
q = video
k = v = text

# Cross-Attention
attented = cross-attention(
    query = q,
    key = k,
    value = v
)

# LayerNorm and Residual Connection
result = layer_norm(attented + q)
```

需要注意的是：这里的 **残差连接（residual connection）**添加的是 **Query 向量**，而非 Key 或者 Value 向量。

## 图示

在论文 [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) 中给出了标准 Transformer 的结构示意图，如下所示：

![标准 Transformer 的结构示意图](/images/fragments/transformer-architecture.png)


在论文 [Multimodal Learning with Transformers: A Survey](https://arxiv.org/abs/2206.06488) 中给出了几种不同的模态融合方式（其中就包括了 Cross-Attention），如下图所示：

![几种不同的模态融合方式](https://pic4.zhimg.com/80/v2-4f55a9338e459570e6d3079d26219c07_720w.webp)


## 参考

- 论文：

  - [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

  - [Multimodal Learning with Transformers: A Survey](https://arxiv.org/abs/2206.06488)

