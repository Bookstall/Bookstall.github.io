---
layout: fragment
title: Gemini 技术报告
tags: [LLM]
excerpt: Gemini 技术报告
keywords: LLM
mathjax: true
---

## Gemini 技术报告

在技术报告中，谷歌表示 Gemini 是一个多模态大模型体系，它在图像、音频、视频和文本理解方面表现出卓越的能力。Gemini 系列包括 Ultra、Pro 和 Nano 三个版本，适用于从复杂推理任务到移动设备的各种应用。

通过在大量基准的跑分表明，功能最强大的 Gemini Ultra 在 32 个基准中的 30 个中刷新了 SOTA（业内最佳）水平。谷歌特别指出，Gemini 是第一个在经过充分研究的考试基准 MMLU 上实现人类专家表现的模型。谷歌相信，Gemini 在跨模态推理和语言理解方面的突出能力将支持各种用例。

Gemini 1.0 有三种尺寸 Ultra 、 Pro 以及 Nano ，如下所示：

- Ultra：可以在各种高度复杂的任务中提供 SOTA 性能，包括推理和多模态任务。它还可以在 TPU 加速器上有效地进行大规模服务；

- Pro：是谷歌在成本和延迟方面进行性能优化的模型，可在各种任务中提供良好的性能，并表现出强大的推理性能和广泛的多模态能力；

- Nano：谷歌最高效的模型，专为在设备上运行而设计。采用**知识蒸馏**的方式，以 Gemini Ultra 和 Gemini Pro 为 **教师模型**，谷歌训练了两个版本的 Nano，参数分别为 1.8B (Nano-1) 和 3.25B (Nano-2)，分别针对低内存和高内存设备，采用 **4 位量化** 进行部署，并提供一流的性能。

## 港中文测评报告

![](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/main/images/gemini_vs_gpt.png)


## 参考

- 机器之心：[谷歌Gemini技术报告出炉，作者多达900余人](https://www.jiqizhixin.com/articles/2023-12-21-4)

- 量子位：[谷歌Gemini扳回一局！多模态能力和GPT-4V不分伯仲｜港中文128页全面测评报告](https://www.qbitai.com/2023/12/108550.html)

- 技术报告：[Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805)

- 测评报告：[A Challenger to GPT-4V? Early Explorations of Gemini in Visual Expertise](https://arxiv.org/abs/2312.12436)

  - [Github 仓库：Awesome-Multimodal-Large-Language-Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)