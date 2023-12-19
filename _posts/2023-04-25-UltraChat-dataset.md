---
layout: post
author: bookstall
tags: 工具, 数据集
categories: [工具, 数据集]
excerpt: 清华大学 开源了 UltraChat 数据集，有两个独立 ChatGPT Turbo API 的对话数据组成
keywords: 工具, 数据集
title: UltraChat 数据集：两个独立 ChatGPT Turbo API 的对话
mathjax: true
---

## UltraChat 数据集：两个独立 ChatGPT Turbo API 的对话

### 背景

自 ChatGPT 发布以来，这段时间对话模型的热度只增不减。当我们赞叹这些模型表现惊艳的同时，也应该猜到其背后巨大的算力和海量数据的支持。

单就数据而言，**高质量的数据至关重要**，为此 OpenAI 对数据和标注工作下了很大力气。有多项研究表明，ChatGPT 是比人类更加可靠的数据标注者，如果开源社区可以获得 ChatGPT 等强大语言模型的大量对话数据，就可以训练出性能更好的对话模型。这一点羊驼系列模型（Alpaca、Vicuna、Koala）已经证明过。例如，Vicuna 使用从 ShareGPT 收集的用户共享数据对 LLaMA 模型进行指令微调，就复刻了 ChatGPT 九成功力。越来越多的证据表明，数据是训练强大语言模型的第一生产力。

ShareGPT 是一个 ChatGPT 数据共享网站，用户会上传自己觉得有趣的 ChatGPT 回答。ShareGPT 上的数据是开放但琐碎的，需要研究人员自己收集整理。如果能够有一个高质量的，覆盖范围广泛的数据集，开源社区在对话模型研发方面将会事半功倍。

基于此，最近一个名为 UltraChat 的项目就系统构建了一个 **超高质量的对话数据集**。项目作者尝试**用两个独立的 ChatGPT Turbo API 进行对话**，从而生成多轮对话数据。

### UltraChat

> - github repo：[UltraChat](https://github.com/thunlp/UltraChat)


![UltraChat logo](https://image.jiqizhixin.com/uploads/editor/715a3274-5a0a-4bcc-b09f-69cb66a1b043/640.png)

UltraChat 旨在构建一个开源、大规模、多轮的基于 Turbo APIs 的对话数据，方便研究者开发具有通用对话能力的强大语言模型。此外，考虑到隐私保护等因素，该项目不会直接使用互联网上的数据作为提示。为了确保生成数据质量，研究者在生成过程中采用了两个独立的 ChatGPT Turbo API，其中一个模型扮演用户角色来生成问题或指令，另一个模型生成反馈。

通过精心设计的提示来指导用户模型模仿人类用户行为并迭代调用这两个 API，生成的对话经过进一步的后处理和过滤。

#### 数据集的组成

如果直接使用 ChatGPT 基于一些种子对话和问题让其自由生成，容易出现话题单一、内容重复等问题，从而难以保证数据本身的多样性。为此，UltraChat 对对话数据覆盖的主题和任务类型进行了系统的分类和设计，还对用户模型和回复模型进行了细致的提示工程，它包含三个部分：

- 关于世界的问题（Questions about the World）：这部分对话来自于对现实世界中的概念、实体和对象相关的广泛询问。所涉及的主题涵盖科技、艺术、金融等多个领域。

- 写作与创作（Writing and Creation）：这部分对话数据着重于指示 AI 从头进行创作一个完整的文本材料，并在此基础上进行后续的提问或进一步指导以完善写作，撰写的材料内容类型包括文章、博客、诗歌、故事、戏剧，电子邮件等等。

- 对于现有资料的辅助改写（Writing and Creation）：该对话数据是基于现有资料生成的，指令包括但不限于改写、续写、翻译、归纳、推理等，涵盖主题同样非常多样。

#### 数据集的构造

这三部分数据覆盖了大部分用户对于 AI 模型的要求。同时，这三类数据也会面临着不同的挑战，为此需要不同的构造方法。

例如，第一部分的数据主要挑战在于如何在总量为几十万组对话中尽量广泛地涵盖人类社会中的 **常见知识**，为此研究者从自动生成的主题和来源于 Wikidata 的实体两个方面进行了筛选和构造。

第二、三部分的挑战主要来自于如何模拟用户指令，并在后续对话中让用户模型的生成尽量多样化的同时又不偏离对话的最终目标（按照要求生成材料或改写材料），为此研究者对用户模型的输入提示进行了充分的设计和实验。在构造完成之后，作者还对数据进行了后处理以削弱幻觉问题。

目前，该项目已经发布了前两部分的数据，数据量为 124 万条，应该是目前开源社区内规模最大的相关数据集。内容包含在现实世界中丰富多彩的对话，最后一部分的数据以及中文版本的数据将在未来发布。

世界问题数据来源于 30 个具有代表性和多样性的元主题，如下图所示：

![](https://image.jiqizhixin.com/uploads/editor/702f2363-100f-4602-87b0-32f4700b96a5/640.png)

- 基于以上元主题，该项目生成了 1100 + 子主题用于数据构建；

- 对于每个子主题，最多生成 10 个具体问题；

- 然后使用 Turbo API 为 10 个问题中的每一个生成新的相关问题；

- 对于每个问题，如上所述迭代地使用两个模型生成 3~7 轮对话。

此外，该项目从维基数据中收集了最常用的 10000 个命名实体；使用 ChatGPT API 为每个实体生成 5 个元问题；对于每个元问题，生成 10 个更具体的问题和 20 个相关但一般的问题；采样 20w 个特定问题和 25w 个一般问题以及 5w 个元问题，并为每个问题生成了 3~7 轮对话。


#### 预览数据集

作者提供了相应的预览地址：

- [预览](http://39.101.77.220/)

- [Atlas 预览](https://atlas.nomic.ai/map/0ce65783-c3a9-40b5-895d-384933f50081/a7b46301-022f-45d8-bbf4-98107eabdbac)

![在 Atlas 中预览 UltraChat 数据集](https://image.jiqizhixin.com/uploads/editor/56690e32-4397-43ab-9c86-c20fc232a9af/640.gif)


#### 下载数据集

> 目前的数据集大小有 **9 GB**

- 通过 [HuggingFace 下载](https://huggingface.co/datasets/stingning/ultrachat)

- 直接下载（清华云）

  - [Questions about the World [Part I + Part II]](https://cloud.tsinghua.edu.cn/f/0a27393192ad46a5a081/?dl=1)

  - [Writing and Creation [Part I]](https://cloud.tsinghua.edu.cn/f/57258a87846243218a9b/?dl=1)

  - [Writing and Creation [Part II]](https://cloud.tsinghua.edu.cn/f/099b4dd71b82448fb7fb/?dl=1)

  - [Assistance on Existent Materials [Part I]](https://cloud.tsinghua.edu.cn/f/1f7abdf2d2564cb4b338/?dl=1)

#### 数据集的格式

下载的数据文件中的每一行都是一个 json 字典，其中包含列表格式的数据 ID 和对话数据。下面是一个示例行。

```json
{
  "id": "0", 
  "data": [
    "How can cross training benefit groups like runners, swimmers, or weightlifters?", 
    "Cross training can benefit groups like runners, swimmers, or weightlifters in the following ways: ...", 
    "That makes sense. I've been wanting to improve my running time, but I never thought about incorporating strength training. Do you have any recommendations for specific exercises?", 
    "Sure, here are some strength training exercises that can benefit runners: ...", 
    "Hmm, I'm not really a fan of weightlifting though. Can I incorporate other forms of exercise into my routine to improve my running time?", 
    "Yes, absolutely! ...",
    "..."
    ]
}
```

更直观的形式可以在 [HuggingFace](https://huggingface.co/datasets/stingning/ultrachat) 上查看。


### 使用 UltraChat 微调 LLaMA-7B

[机器之心](https://www.jiqizhixin.com/articles/2023-04-21-11) 的作者尝试使用开源的 LLaMa-7B 模型在 UltraChat 上进行监督的指令微调，发现仅仅训练 10000 步后就有非常可观的效果，一些例子如下：

> 世界知识：分别列出 10 个很好的中国和美国大学

![10 个很好的中国和美国大学](https://image.jiqizhixin.com/uploads/editor/02bc237f-d109-4acd-a554-64a0f7e714d1/640.png)

> 想象问题：当时空旅行成为可能后，有什么可能的后果？

![当时空旅行成为可能后，有什么可能的后果？](https://image.jiqizhixin.com/uploads/editor/1376a65c-8ac2-4c7b-8360-e8dab9c75a47/640.png)

> 三段论：鲸鱼是鱼吗？

![鲸鱼是鱼吗？](https://image.jiqizhixin.com/uploads/editor/1122544e-8f4d-4515-b682-9ee740383432/640.png)

LLaMA-7B 给出的回答是：鲸鱼不是鱼，而是海洋中的一种哺乳动物。
> 假设问题：证明成龙比李小龙更出色

![证明成龙比李小龙更出色](https://image.jiqizhixin.com/uploads/editor/8a7bf6ee-3c45-420f-b90e-ac84a04c2495/640.png)

LLaMA-7B 给出的回答是：两人都很优秀，对两人的评价因人而异。

## Atlas

> Explore, label, search and share massive datasets in your web browser.
>
> 在您的 Web 浏览器中探索、标记、搜索和共享海量数据集

### 背景

传统设计工具以设计师构建完成的稿纸与设计师明确的产出预期为导向，一步步趋近设计结果。生成式设计工具以巨量数据集作为支撑，以设计命题牵引出多种可能的设计结果，供设计师选择。

生成式设计工具对于数据集的硬性需求，产生了许多以数据信息检索、查询、整合为主要功能的产品，如AI生成内容搜索引擎 lexica.art 。ATLAS 则更进一步，锚准了 AI 生成数据集领域，推出百万量数据的 explorable map of KREA AI's Stable Diffusion Search Engine 图谱式搜索引擎项目。

---

**Nomic AI** 是世界上第一家信息制图公司，[Atlas](https://atlas.nomic.ai/) 是其旗下产品，**支持用户轻松查看、探索大量非结构化数据并与之交互**。

[An explorable map of KREA AI's Stable Diffusion Search Engine](https://atlas.nomic.ai/map/809ef16a-5b2d-4291-b772-a913f4c8ee61/9ed7d171-650b-4526-85bf-3592ee51ea31) 是 Atlas 最近新推出的代表性数据集项目，目前已整合了 600w+量级的 Stable Diffusion 生成图像的数据。

同时，Nomic AI 还开源了最近很火的 [gpt4all](https://github.com/nomic-ai/gpt4all)。

### 功能

- 整体浏览图谱

- 支持缩放图谱

- 点击节点查看

- 搜索关键词

- 支持节点筛选模式与颜色等细节调整

### 更多

- [Atlas 文档](https://docs.nomic.ai/how_does_atlas_work.html)

- [Atlas Colab Demo](https://colab.research.google.com/drive/1bquOLIaGlu7O_CFc0Wz74HITzWs4UEa4?usp=sharing)


## 参考

- 机器之心：[调用多个ChatGPT API相互对话，清华开源的多轮对话数据UltraChat来了](https://www.jiqizhixin.com/articles/2023-04-21-11)

- github repo：[UltraChat](https://github.com/thunlp/UltraChat)

- CSDN：[创作没灵感？可视化图谱+搜索引擎助你无障碍生成内容 #ATLAS + Stable Diffusion](https://blog.csdn.net/shadowcz007/article/details/127662216)

- github repo：[gpt4all](https://github.com/nomic-ai/gpt4all)

