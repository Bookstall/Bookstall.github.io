---
layout: post
author: bookstall
tags: Prompt Tuning, Multimodal
categories: [Prompt Tuning, Multimodal]
excerpt:  Prompt Tuning for Generative Multimodal Pretrained Models
keywords: Prompt Tuning, Multimodal
title: 论文：Prompt Tuning for Generative Multimodal Pretrained Models
mathjax: true
---


## 前言

> 《Prompt Tuning for Generative Multimodal Pretrained Models》
>
> - URL：https://arxiv.org/abs/2208.02532
>
> - 单位：阿里达摩院
>
> - Code：https://github.com/OFA-Sys/OFA/blob/main/prompt_tuning.md

In this work, we explore **prompt tuning for generative multimodal pretrained models**. Through extensive experiments, we demonstrate that the  light-weight prompt tuning can **achieve comparable performance** with finetuning with much fewer parameters to tune (e.g., 1%), and it can <u>surpass other  light-weight tuning methods</u>, e.g., <u>Adapter</u> and <u>BitFit</u>. 

Through our analysis, we figure out **a significant advantage of prompt tuning about its robustness against adversarial attack**. Furthermore, we provide a comprehensive analysis about the influence of prompt tuning setups, including the **prompt length, prompt depth, and reparameterization**. Potentially prompt tuning can be an alternative to  finetuning, but still, there are **some salient limitations** in this method, e.g., **slow convergence and  training instabilities**. We hope that future studies in this field can alleviate the aforementioned problems and thus promote the application of prompt tuning.


## 整体结构

![Prompt Tuning OFA Model 的整体结构](/images/posts/Prompt-Tuning-OFA/prompt-tuning-OFA-framework.png)

> The cross-attention is essentially **multi-head attention**, where **the keys $$K$$ and values $$V$$ are the transformation of the encoder output states**, instead of the inputs.



> 这里默认使用的是 **prefix-tuning** 的方式，将 prompt 插入到 prefix。其他的插入方式 middle、end 效果差不多，因此，作者这里就直接使用了 prefix-tuning。



## 实验



### 1）与 Finetuning 进行比较



### 2）与其他 efficient tuning methods 进行比较

![与其他微调方法的比较](/images/posts/Prompt-Tuning-OFA/prompt-tuning-OFA-result-in-different-efficient-tuning-methods.png)

### 3）鲁棒性分析



### 4）消融实验

#### Prompt Length 的影响

使用了 $$\{10, 16, 30, 64, 100, 120\}$$ 这几种不长度的 prompt 序列。

> 论文默认使用的 Prompt Length = 64（最佳性能）

![Prompt Length 的消融实验-1](/images/posts/Prompt-Tuning-OFA/prompt-tuning-OFA-result-in-prompt-length-1.png)

![Prompt Length 的消融实验-2](/images/posts/Prompt-Tuning-OFA/prompt-tuning-OFA-result-in-prompt-length-2.png)


#### Prompt Depth 的影响

将 Prompt 插入到三个不同的位置，包括：Encoder-Only、Decoder-Only 以及 Encoder-Decoder。

![Prompt Depth 的消融实验](/images/posts/Prompt-Tuning-OFA/prompt-tuning-OFA-result-in-prompt-depth.png)

#### Reparameterization 的影响

根据经验，**直接更新可训练嵌入** 会导致 <u>不稳定的训练</u> 和 <u>性能的轻微下降</u>。之前的工作通常利用编码器（例如 MLP）来**重新参数化可训练的嵌入**。





## 讨论

尽管 Prompt Tuning 有很多的优点，但是仍然无法完全替代 Fine-Tuning。主要的局限性包括：

- 收敛速度慢（slow convergence）

  - 需要 <u>更多的 epochs</u>，才能达到 Fine-Tuning 同等的性能

  - 虽然 Prompt-Tuning 的训练效率高，但是 <u>更多的训练 epochs 也可能导致更多的计算成本</u>

  - 这也表明：除了达到与 Fine-Tuning 相当甚至改进的性能外，能够快速、稳定收敛的方法也很重要（速度与稳定之间的 trade-off）

- 难以找到合适的超参数设置（suitable hyperparameter setup）

Despite the aforementioned limitations, **prompt tuning demonstrates significantly better robustness against adversarial attack**.

尽管存在上述局限性，prompt tuning 能够 **更好的防御对抗攻击**，提高模型的鲁棒性。


## 参考

- 论文：[Prompt Tuning for Generative Multimodal Pretrained Models](https://arxiv.org/abs/2208.02532)



