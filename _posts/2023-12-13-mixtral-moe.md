---
layout: post
author: bookstall
tags: Transformer
categories: [LLM, MoE]
excerpt: Mixtral 8x7B
keywords: LLM, MoE
title: Mixtral 8x7B：首个开源 MoE 大模型
mathjax: true
sticky: true
---


## Mixture of Experts（MoE）

> 参考：[Mixture of Experts Explained](https://huggingface.co/blog/moe)


![MoE layer from the Outrageously Large Neural Network paper](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/01_moe_layer.png)

![MoE Transformer Encoder from the GShard Paper](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/02_moe_block.png)

![Switch Transformer Layer of the Switch Transformer paper](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/03_switch_layer.png)

![Mixtral 7B's MoE](https://github.com/mistralai/mistral-src/raw/main/assets/smoe.png)

## Mixtral

### 基本超越 LLaMA 2 70B

> **Mixtral 8×7B** 在大多数基准测试中都优于 LLaMA2 70B，推理速度快了 6 倍。
> 
> 它是最强大的、具有宽松许可的开放权重模型，也是最佳性价比之选。

具体来说，Mixtral 采用了稀疏混合专家网络，是一个 **decoder-only** 的模型。其中，前馈块（FFN）会从 8 组不同的参数组中进行选择。也就是说，实际上，Mixtral 8×7B 并不是 8 个 7B 参数模型的集合，仅仅是 Transformer 中的前馈块（FFN）有不同的 8 份。

这也就是为什么 Mixtral 的参数量并不是 56B，而是 46.7B。

其特点包括以下几个方面：

- 在大多数基准测试中表现优于 Llama2 70B，甚至足以击败 GPT-3.5

- 上下文窗口为 32k

- 可以处理英语、法语、意大利语、德语和西班牙语

- 在代码生成方面表现优异

- 遵循 Apache 2.0 许可（免费商用）

具体测试结果如下：

![](https://img.ithome.com/newsuploadfiles/2023/12/e9cdfb69-a78e-49fd-b47c-76be2b6455c3.png?x-bce-process=image/quality,q_75/format,f_webp)

另外，在幻觉问题方面，Mixtral 的表现也优于 LLaMA2 70B：

- 在 TruthfulQA 基准上的成绩是 73.9% vs 50.2%；

- 在 BBQ 基准上呈现更少的偏见；

- 在 BOLD 上，Mixtral 显示出比 LLaMA2 更积极的情绪。

此次与 Mixtral 8×7B 基础版本一起发布的，还有 Mixtral 8x7B Instruct 版本。后者经过 SFT 和 DPO 优化，在 MT-Bench 上拿到了 8.3 的分数，跟 GPT-3.5 差不多，优于其他开源大模型。

![](https://img.ithome.com/newsuploadfiles/2023/12/9442a6ab-fb3a-48dc-bcca-fb0d1cc930ae.png?x-bce-process=image/quality,q_75/format,f_webp)

---

更关键的是，普林斯顿博士生 Tianle Cai分析了 Mistral-7B 与 Mixtral-8x7B 模型的权重相关性做了分析，证明了模型的成功复用。

随后网友发现，Mistral AI 创始人也亲自证实，**MoE 模型确实就是把 7B 基础模型复制 8 次，再进一步训练来的**。

![](https://www.qbitai.com/wp-content/uploads/replace/8e2d4d845abde9dc8e74c07e1aa996ea.png)

### 体验

目前，Mistral 官方已经宣布上线 API 服务，不过还是邀请制，未受邀用户需要排队等待。

值得关注的是，API 分为三个版本：

- 小小杯（Mistral-tiny）：对应模型是 Mistral 7B Instruct；

- 小杯（Mistral-small）：对应模型是这次发布的 Mixtral 8×7B；

- 中杯（Mistral-medium）：对应的模型尚未公布，但官方透露其在 MT-Bench 上的得分为 8.6 分。

而在线版本，目前还只能到第三方平台（Poe、HuggingFace等）体验。

### 能看懂中文，但不太愿意说

虽然官方通告中并没有说支持中文，但我们实测（**[HuggingFace Chat](https://huggingface.co/chat/?model=mistralai/Mixtral-8x7B-Instruct-v0.1) 中的在线版，模型为 Instruct v0.1 版本**）发现，Mixtral 至少在理解层面上已经具备一定中文能力了。

生成层面上，Mixtral 不太倾向于用中文来回答，但如果指明的话也能得到中文回复，不过还是有些中英混杂的情况。

![](https://www.qbitai.com/wp-content/uploads/replace/7c58b85668550b59ceb1d16821689455.png)


## 参考

- 量子位：

  - [开源大模型超越 GPT-3.5！爆火 MoE 实测结果出炉，网友：OpenAI 越来越没护城河了](https://www.qbitai.com/2023/12/105808.html)

  - [首个开源 MoE 大模型发布！GPT-4 同款架构，来自欧洲的 OpenAI](https://www.qbitai.com/2023/12/105154.html)


- Mistral
  
  - 博文：[Mixtral of experts](https://mistral.ai/news/mixtral-of-experts/)

  - GitHub：https://github.com/mistralai/megablocks-public

- HuggingFace：

  - 博文：[Mixture of Experts Explained](https://huggingface.co/blog/moe)

  - 博文：[Welcome Mixtral - a SOTA Mixture of Experts on Hugging Face](https://huggingface.co/blog/mixtral)

- Mistral 7B
  
  - 论文：[Mistral 7B](https://arxiv.org/abs/2310.06825)

  - 博文：[Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/)

  - 代码：https://github.com/mistralai/mistral-src

