---
layout: post
author: bookstall
tags: LLM, Transformer
categories: [LLM, Transformer]
excerpt: 整理了几个关于 Transformer 上下文长度的工作
keywords: LLM, Transformer
title: 更长上下文的 Transformer
mathjax: true
---

## 前言



## Transformer

![Transformer Encoder 结构示意图](https://img-blog.csdnimg.cn/20190407095343453.png)



## Vanilla Transformer

> 为何要提这个模型？因为 Transformer-XL 是基于这个模型进行的改进。

![Vanilla Transformer 预测任务的示意图](https://img-blog.csdnimg.cn/20190407095512873.png)




## Transformer-XL

Transformer-XL 架构在 Vanilla Transformer 的基础上引入了两点创新：**循环机制（Recurrence Mechanism）** 和 **相对位置编码（Relative Positional Encoding）**，以克服 Vanilla Transformer 的缺点。

与 Vanilla Transformer 相比，Transformer-XL 的另一个优势是它可以被用于单词级和字符级的语言建模。

### 循环机制

![循环机制的示意图](https://img-blog.csdnimg.cn/20190407095601191.png)


### 相对位置编码


## Lightning Attention-2




## Infini-Transformer

![Infini-Transformer 示意图](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/fd13c9177f83462f87c1354466e959a4~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=858&h=1148&s=68904&e=webp&b=fcfafa)


标准的 Transformer（注意力机制）不会保留上下文信息，会将上下文信息直接丢弃掉。也就是说，对于每一个 segment，都是用一个对应的 Transformer block 来进行处理。

Transformer-XL 对于 segment 1 仅使用一个 transformer block，而对于segment 2 使用前两个 transformer block，这是因为 Transformer-XL 采用了一种特殊的机制来处理长序列数据，这种机制被称为 “segment-level recurrence”。

- Transformer-XL 通过重用之前处理过的 segment 的隐藏状态（即 key-value 对），来增强当前 segment 的上下文信息。这样做的目的是为了解决在处理长序列时出现的上下文碎片化问题，即在标准的 Transformer 模型中，每个 segment 的处理是独立的，不包含之前 segment 的信息，这限制了模型捕获长期依赖的能力。

- 在图二中，segment 1 是序列的第一个 segment，因此它只使用了一个 transformer block 来处理当前的序列数据。然而，当处理 segment 2 时，Transformer-XL 会将 segment 1 中计算出的隐藏状态缓存起来，并在处理 segment 2 时将这些状态作为额外的上下文信息一起考虑。这就解释了为什么在处理 segment 2 时，Transformer-XL 会使用前两个transformer block：第一个 block 用于处理 segment 1，而第二个 block 则同时处理 segment 2 和重用的 segment 1 的隐藏状态。

- 这种设计允许 Transformer-XL 在处理每个新的 segment 时，都能够访问到之前的上下文信息，从而有效地扩展了模型的上下文窗口，使得模型能够捕获更长序列中的依赖关系。这种方法显著提高了模型在处理长序列时的性能，尤其是在语言建模和其他需要长期依赖建模的任务中。

当 Transformer-XL 处理第 N 个 segment 时，它会使用 N 个 Transformer block。每个 block 处理一个segment的输入数据，同时还会重用前 N-1 个 segments 的隐藏状态（即 key-value 对）。这些重用的隐藏状态被缓存并作为额外的上下文信息传递给当前正在处理的 segment，以帮助模型捕获长期的依赖关系。

确实，保存和加载这些重用的隐藏状态需要一定的内存或存储空间。Transformer-XL 通过使用 segment-level recurrence 机制来缓存之前segments的隐藏状态，这会增加内存消耗。然而，相比于直接处理整个长序列，这种方法仍然显著减少了所需的内存，因为它不需要存储整个长序列的 key-value 对，而是只存储每个 segment 的隐藏状态。

![Infini-Transformer 与 Transformer-XL 的比较](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/85d14eeecc2d44c9a34e4b78f1a45725~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=1080&h=626&s=36598&e=webp&b=f3f2f2)


> Infini-Transformer 与 Transformer-XL 的方法区别和联系：
> 
> - **上下文管理**: Transformer-XL 通过缓存前一个 segment 的隐藏状态来扩展上下文，而 Infini-Transformer 使用压缩记忆来存储长期依赖信息，这样可以在处理新的 segments 时重用这些信息。
> 
> - **内存效率**: Transformer-XL 随着序列长度的增加，需要缓存更多的隐藏状态，这可能导致内存消耗增加。Infini-Transformer 通过压缩记忆机制，使用固定数量的参数来存储长期依赖，从而提高了内存效率。
>
> - **注意力机制**: Infini-Transformer 引入了线性注意力（linear attention）机制，这是一种基于关联矩阵的注意力机制，它允许模型以更高效的方式更新和检索压缩记忆中的信息。

---

与多头注意力（MHA）类似，除了点积注意力之外，Infini-attention 还为每个注意力层维护 H 个并行压缩内存（H 是注意力头的数量）。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/49bce75eb05145e4ba2bcbd9bcb91b25~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=876&h=1232&s=103454&e=webp&b=fcf7f7)

---

下表 1 列出了几种模型根据模型参数和输入 segment 长度，定义的上下文内存占用和有效上下文长度。Infini-Transformer 支持具有有限内存占用的无限上下文窗口。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b5a42f83a9b549479522ee1d8b8031c9~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=1080&h=342&s=67440&e=webp&b=fbfbfb)



### 关键技术

压缩记忆（Compressive Memory）

- 压缩记忆是一种参数化函数，用于存储和检索信息。它不同于传统的变换器中随输入序列长度增长的 KV（Key-Value）记忆数组，压缩记忆系统保持固定数量的参数来存储信息，以实现计算效率。

- 在 Infini-attention 中，压缩记忆通过重用点积注意力计算中的查询（Q）、键（K）和值（V）状态，而不是像传统注意力机制那样丢弃它们。这样，新的信息可以通过改变这些参数来添加到记忆中，并且可以在后续处理中被检索出来。

线性注意力（Linear Attention）

- 线性注意力是一种计算效率更高的注意力机制，它通过简化注意力计算来降低内存占用和计算时间。在 Infini-attention 中，线性注意力用于处理无限长上下文，并且与压缩记忆结合使用。

- 通过使用关联矩阵（associative matrix）和线性更新规则，Infini-attention 能够在保持训练稳定性的同时，实现对长期记忆的高效检索和更新。

分段处理（Segment-based Processing）

- 为了处理无限长的输入序列，Infini-attention 采用了分段处理的方法。每个输入序列被分割成多个较短的段（segments），每个段都在局部上下文中进行处理。

- 通过这种方式，模型可以在保持固定长度的局部注意力窗口的同时，通过递归地处理每个段来处理整个无限长的序列。

长期和局部信息的融合

- Infini-attention 通过一个学习到的门控标量（gating scalar）β来融合局部注意力状态和从压缩记忆中检索到的内容。

- 这允许模型在长短期信息流之间进行可学习的权衡，同时保持对当前上下文的敏感性和对长期依赖的记忆。


### 优势

处理长序列的能力：

- Infini-attention 模型能够处理无限长的输入序列，这对于需要理解和生成长文本的应用场景非常有用，如文档摘要、对话系统和文本生成等。

内存和计算效率：

- 通过压缩记忆和线性注意力机制，Infini-attention 在处理长序列时能够保持有界的内存和计算资源，这对于部署在资源受限的环境中尤其重要。

长短期信息的融合：

- 模型能够有效地结合长短期信息，这有助于在保持对当前上下文敏感的同时，也能够考虑到长期依赖关系。

持续预训练和适应性：

- Infini-attention 支持持续预训练和长上下文适应，这意味着模型可以通过持续学习来改进其性能，并适应新的数据和任务。

性能提升：

- 在长上下文语言建模和特定任务（如密钥检索和书籍摘要）中，Infini-attention 模型展现出了优于现有技术的性能。

---

Infini-Transformer的优势：

- **无限上下文**: Infini-Transformer 能够处理无限长的输入序列，而不会受限于固定长度的上下文限制。

- **内存和计算效率**: 由于使用了压缩记忆，Infini-Transformer 能够在保持长期依赖信息的同时，减少内存和计算资源的需求。

- **流式处理**: Infini-Transformer 支持流式处理长序列数据，这意味着它可以逐步处理输入数据，而不需要一次性加载整个序列。

- **适应性**: Infini-Transformer 的设计允许模型通过持续预训练和任务微调来适应长上下文，这使得模型能够在处理长序列任务时表现出色。


### 局限性

复杂性：

- 虽然 Infini-attention 提高了效率，但其引入的压缩记忆和线性注意力机制可能增加了模型的复杂性，这可能需要更多的研发努力来实现和优化。

训练稳定性：

- 压缩记忆的更新和检索过程可能需要精心设计的训练策略来保证稳定性，特别是在面对非常长的序列时。

泛化能力：

- 尽管 Infini-attention 在特定任务上表现出色，但其在不同类型的任务和数据集上的泛化能力还需要进一步验证。

实现细节：

- 文档中提到的方法可能依赖于特定的实现细节，这**可能在不同的框架或硬件平台上难以复制**。

资源需求：

- 尽管内存和计算资源得到了优化，但训练和运行如此大型的模型仍然需要相对较高的计算资源。

### 代码

> 参考 Unofficial Implementation：[InfiniTransformer](https://github.com/Beomi/InfiniTransformer)

使用 Google Gemma 模型（`gemma-2b`），配置如下：

```python
config = GemmaConfig.from_pretrained(
    "google/gemma-2b",
    attn_implementation="eager",
)
config.memory_size = 2048
config.use_cache = False
config.segment_size = 16
```

关于 Gemma 模型，可以阅读 [HuggingFace 关于 Gemma 的博客](https://github.com/huggingface/blog/blob/main/zh/gemma.md)。




## 综述

> - 论文首先分析了使用当前基于 Transformer 的模型处理长上下文输入和输出的问题。
> 
> - 然后，提供了一个全面的分类体系，以指导 Transformer 架构升级的领域，来解决这些问题。作者对长上下文 LLM 广泛使用的评估需求进行了调研，包括数据集、度量标准和基准模型，以及一些令人惊奇的优化工具包，如库、系统和编译器，以增强 LLM 在不同阶段的效率和功效。
> 
> - 最后，文章进一步讨论了这一领域未来研究的主要挑战和潜在方向。
> 
> 作者还建立了一个 [Github 仓库](https://github.com/Strivin0311/long-llms-learning)，汇总了相关文献，并提供实时更新。

文章从基本的语言建模目标开始，内容涵盖从典型的建模阶段到在基于 Transformer 的仅解码 LLM 中找到的关键架构模块，如下图 (a) 所示。随后，作者对 LLM 在遇到扩展上下文窗口时的架构限制进行了简要分析。最后提出了一个全面的方法论分类法，旨在通过架构创新增强 LLM 的长上下文能力，如下图 (b) 所示。

![长上下文的 Tansformer 结构和方法示意图](https://github.com/Strivin0311/long-llms-learning/raw/main/imgs/overview_with_caption_v2.png)

其中，

- (a) 现代基于 Transformer 的仅解码 LLMs 的典型架构解剖图，右上角有图例；

- (b) 用于增强 Transformer 架构模块的方法论分类法（与 (a) 相对应的颜色），包括：

  - 高效注意力（注意力核心的子模块）
  
  - 长期记忆（针对 KV 缓存）
  
  - 外推性 PEs（针对位置嵌入模块）
  
  - 上下文处理（与上下文预 / 后处理有关）
  
  - 杂项（整个解码器块以及损失模块通用）

### 难点

- **注意力机制的复杂度**：

  - 随着序列长度的增加，时间和空间计算成本都呈 **二次方** 增加，这对于训练和推理可能都是繁重的

- **上下文记忆**：LLM 缺乏显式的记忆机制，完全依赖 KV 缓存来存储列表中所有先前 token 的表示

- **最大长度的约束**：

  - 在训练阶段，工程师通常需要确定一个关键的超参数 $$L_{max}$$

  - 在推理阶段，LLM 的服务提供者还必须限制用户提示的长度或自动截断它们以与预定义的 $$L_{max}$$ 对齐，即使推理资源通常比训练阶段更丰富

  - 理论上 **只要资源足够**，Transformer 可以处理任意长度的序列。然而，当前的语言模型在处理超过 $$L_{max}$$ 的输入序列时通常表现出明显的性能下降，经常导致重复和不切实际的输出。


### 改进的新方法

**高效的注意力**：这些方法侧重于实现具有降低计算要求的高效注意力机制，甚至实现了线性复杂度。通过这样做，它们能够通过直接在预训练阶段增加 $$L_{max}$$ 来推进 LLM 在推理期间的有效上下文长度边界。

**长期记忆**：为了解决上下文工作记忆的局限性，一些方法旨在设计明确的 **记忆机制**，弥补 LLM 中缺乏高效和有效的长期记忆的不足。

**具备外推性的位置编码**：通过改进现有位置编码方案的外推性能来增强 LLM 的长度泛化能力。

**上下文处理**：除了增强特定低级 Transformer 模块的方法外，一些方法涉及对现成的 LLM 与额外的上下文预 / 后处理。这些方法确保每次调用 LLM 时输入始终满足最大长度要求，并通过引入多个调用开销打破上下文窗口限制。

**杂项**：探讨了各种一般且有价值的方法，这些方法不容易归入前面四类，为推进 LLM 的长上下文能力提供了更广泛的视角。


### 未来方向

注意力 Trade-off

记忆效果和效率

长度外推的挖掘

特定但通用的目标

可靠的度量/评估

## 参考


- Transformer-XL

  - 机器之心：

  - CSDN：[Transformer-XL解读（论文 + PyTorch源码）](https://blog.csdn.net/Magical_Bubble/article/details/89060213)

  - 论文：[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)

  - [Github: transformer-xl](https://github.com/kimiyoung/transformer-xl/)
  

- Lightning Attention-2

  - 机器之心：[新一代注意力机制 Lightning Attention-2：无限序列长度、恒定算力开销、更高建模精度](https://zhuanlan.zhihu.com/p/678552539)

- Infini-Transformer

  - 论文：[Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention](https://arxiv.org/abs/2404.07143)

  - 机器之心：[直接扩展到无限长，谷歌 Infini-Transformer 终结上下文长度之争](https://juejin.cn/post/7357288361235972137)

  - Unoffical Implementation：[InfiniTransformer](https://github.com/Beomi/InfiniTransformer)

- 综述

  - 论文：[Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey](https://arxiv.org/abs/2311.12351)

  - [Github: long-llms-learning](https://github.com/Strivin0311/long-llms-learning)


