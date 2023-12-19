---
layout: wiki
title: 2023-04-17：论文速递
cate1: paper
cate2:
description: Pre Layer Normalization in Transformers
keywords: paper
---

## 2023-04-17：论文速递

### 1、DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning

> DiffFit: 通过简单的参数高效微调解锁大扩散模型可迁移性
> 
> - URL：https://arxiv.org/abs/2304.06648
> 
> - 单位：华为诺亚方舟实验室
>
> - Tech Report（技术报告）

- 动机：虽然扩散模型在生成高质量图像方面表现出色，但将大型预训练扩散模型应用于新领域仍然是一个挑战。本文的动机是开发一种简单且高效的微调策略，以提高扩散模型在实际应用中的适用性。

- 方法：本文提出 DiffFit，一种参数高效的微调策略，仅微调特定层中的**偏置项（bias）**和新添加的**缩放因子（scale factor）**，以快速适应新领域，还提供了直观的理论分析和详细的消融研究，以进一步深入了解这种简单的参数高效微调策略如何快速适应新的分布。

- 优势：DiffFit 可以快速适应新领域，并且仅使用 0.12％ 的可训练参数。在 8 个下游数据集上，DiffFit 表现优异，与现有的微调策略相比具有更高的效率和更好的性能。此外，DiffFit 可以将低分辨率生成模型无缝扩展到高分辨率图像生成，从而实现更好的生成效果和更快的训练时间。

一句话总结：

- 提出一种名为 DiffFit 的参数高效的微调策略，可以快速适应新的领域。该方法仅微调特定层中的偏置项和新添加的缩放因子，而不需要微调所有参数，从而降低了模型存储成本和训练时间，并且能够在多个数据集上实现较好的效果。

![](https://pic1.zhimg.com/80/v2-281da3b76431b83a4b32041fc5e87734_720w.webp)


![](/images/wiki/DiffFit-Architecture.png)


![](/images/wiki/DiffFit-Algorithm.png)


### 2、Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition

> 针对开放式词汇视觉识别的 20000 类提示预训练

> 《Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition》
>
> - URL：https://arxiv.org/abs/2304.04704
>
> - 作者：李沐
> 
> - 单位：Amazon Web Services
>
> - Official Code：https://github.com/amazon-science/prompt-pretraining :star:

- 动机：旨在学习一个通用的提示，涵盖广泛的视觉概念且与任务无关。

- 方法：本文提出 POMP（**P**r**OM**pt **P**re-training）方法，可以学习一个通用的提示，覆盖了超过 2 万个类别的语义信息，可以直接应用于多种视觉识别任务，包括图像分类、语义分割和目标检测。该方法在 ImageNet-21K 数据集上进行预训练，将语义信息压缩到通用提示中，从而提高模型的性能。

- 优势：POMP 方法可以学习到一个通用的提示，覆盖了广泛的视觉概念且与任务无关，可以直接应用于多种视觉识别任务，包括图像分类、语义分割和目标检测，从而提高模型的性能。实验结果表明，POMP 方法在 21 个下游数据集上取得了最先进的性能，例如在 10 个分类数据集上的平均准确率达到了 67.0％，比 CoOp 高 3.1％。

一句话总结:

- 提出一种名为 POMP 的视觉-语言模型预训练方法，可以学习到一个通用的提示（prompt），覆盖了超过 2 万个类别的语义信息，可以直接应用于多种视觉识别任务，包括 **图像分类、语义分割和目标检测**，从而提高模型的性能。

![](https://pic4.zhimg.com/80/v2-5daeb8e8a746ad0b8e379202a109ae97_720w.webp)

![](https://pic3.zhimg.com/80/v2-8d635ff3058d4ef18cbf6423971fb7de_720w.webp)


### 其他

#### Toxicity in ChatGPT: Analyzing Persona-assigned Language Models

> 大型语言模型人格分配的毒性分析

动机：

- 由于大型语言模型在很多服务中都得到了广泛的应用，如医疗、教育和客户服务等，因此对这些系统的安全性有了更高的要求，特别是对一些关键信息需要的用户，如学生或患者与聊天机器人交互。因此，有必要对大型语言模型的能力和局限性进行清晰的认识和评估。

方法：

- 系统评估 CHATGPT 的毒性，尤其是在给模型指定人格特质时，如何影响生成文本的负面情感和歧视性。


优势：

- 通过对 CHATGPT 的评估，发现了其存在的问题，呼吁整个人工智能社区对模型安全性进行评估和改进，以创造更为健壮、安全和可信赖的人工智能系统。

一句话总结:

- 通过对大型语言模型 CHATGPT 的研究，发现在给模型指定人格特质（如拳击手穆罕默德·阿里）时，生成的文本可能会表现出明显的负面情感和歧视性，对使用者可能会造成潜在的伤害和损失。文章呼吁整个人工智能社区对模型安全性进行评估和改进。

![](https://pic3.zhimg.com/80/v2-ecac2de9cbc337c0657d7d977ce7fcda_720w.webp)

#### 

> 针对 ChatGPT 的多步越狱隐私攻击

动机：

- 由于大型语言模型在许多领域中得到了广泛的应用，因此需要对其集成应用对用户隐私的潜在威胁进行研究，并提出相应的解决方案。

方法：

- 作者对 OpenAI 的模型 API 和 New Bing 进行了隐私威胁研究，揭示了 ChatGPT 可以记住隐私训练数据并提出了一种新的提示方法来暴露用户个人信息。此外，作者还通过实验揭示了 LLM 的隐私风险。

优势：

- 本文是第一个对 LLM 和集成应用的隐私威胁进行全面研究的工作，并提出了相应的解决方案。

一句话总结:

- 研究了大型语言模型（LLM）在集成应用程序中可能带来的隐私威胁，揭示了 ChatGPT 可以记住隐私训练数据，提出一种新的提示方法来暴露用户的个人信息。此外，还揭示了结合搜索引擎使用LLM存在的新的隐私威胁，并提出了相关解决方案。

![](https://pic4.zhimg.com/80/v2-40c886b78e75c6bc82e86261cd86ea4b_720w.webp)

![](https://pic1.zhimg.com/80/v2-c51214ffe950bce1cc5395868c65c890_720w.webp)


## 参考

- 知乎：[爱可可AI前沿推介(4.17)](https://zhuanlan.zhihu.com/p/622430473)

