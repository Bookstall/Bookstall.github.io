---
layout: post
author: bookstall
tags: Transformer
categories: [LLM, video]
excerpt: 清华大学发布 LLM4VG 基准，用于评估 LLM 模型的视频时刻定位能力。
keywords: LLM, video
title: LLM4VG 基准：用于评估 LLM 的视频时刻定位能力
mathjax: true
---

## LLM4VG 结构

### 组成结构

LLM4VG 的完整结构如下图所示，包括以下四个部分：

- Video LLM（VidLLM）：经过视频数据微调的 LLM 模型

- LLM：普通的 LLM 模型

- Video Description Generator：用于生成响应的视频（文本）描述
  
  - Caption-based generator：

  - VQA-based generator：

- Prompt Design：设计了三种 Prompt

  - question prompt

  - description prompt

  - exemplar prompt

![LLM4VG 的组成结构](https://img.ithome.com/newsuploadfiles/2023/12/e2628509-a607-4226-97ed-a74b07384e40.png?x-bce-process=image/format,f_avif)


### 框架

LLM4VG 的完整框架如下图所示：

![LLM4VG 的完整框架](https://img.ithome.com/newsuploadfiles/2023/12/bbada6af-24df-40da-873b-419a8941f204.png?x-bce-process=image/format,f_avif)


使用 Visual Description Generator 生成 **逐秒的视频字幕描述**，以构建 description prompt。

具体来说，LLM4VG 尝试了两种方式生成视频字幕描述：

- **Caption-based generator**：these models would **directly** transform the image into the visual description $$c_i$$ per second.
  
  - Fc：CNN、LSTM

  - Att2in：Attention model
  
  - Updown model
  
  - Transformer：注意力机制
  
  - 高级 caption model：Blip 模型

- **VQA-based generator**：Blip for VQA

  - 考虑到由于基于字幕的生成器的泛化能力较弱，**视觉描述中偶尔会丢失关键信息**（例如，字幕模型提供的许多视觉描述在查询中不包含关键字）

  ```shell
  Question：What is happening in the image?

  Answer：Is it currently happening <query event> in the image（图像中当前正在发生 <query event>）
  ```

---

Prompt Design 使用三种不同的 Prompt：

- **question prompt**：【Zero-shot 的形式】
	
    - Find the start time and end time of the query below from the video

- **description prompt**：【Zero-shot 的形式】

- **exemplar prompt**：【One-shot 的形式】

此外，LLM4VG 还添加了一个 confidence judgment prompt，以便提升模型的定位性能。the one-shot with **confidence judgment prompt** has an additional sentence ‘**judge whether the description sequence is suitable for the video grounding**’。

---

For the result $$\text{Output} = \text{LLM}(\text{Prompt})$$, we will extract the content in the answer for the prediction result of start and  end time. 从 LLM 给出的 answer 中提取开始时间和结束时间的预测结果。


## 实验结果

### 几个结果

- Observation 1. LLMs show preliminary abilities for video grounding tasks, outperforming the VidLLMs.

  - 普通的 LLM > VidLLM

- Observation 2. Different combinations of visual description generators, LLMs, and prompt designs, can significantly affect the recall rate of video grounding

  - visual description generators、LLMs 以及 prompt 都会显著影响 LLM 的定位性能

- Observation 3. LLMs’ ability to complete video grounding tasks not only depends on the model scale but is also related to the models’ ability of handling long sequence  question answers.

  - LLM 完成视频基础任务的能力不仅取决于模型规模，还与模型处理长序列问题答案的能力有关。
   
    - 模型规模：GPT-3.5 > Vicuna-7B / Longchat-7B

    - 处理长序列问题的能力：Longchat-7B > Vicuna-7B

- Observation 4. General advanced caption models as visual description models do not guarantee a performance  boost in helping LLMs conduct video grounding tasks.

 - 通用的高级字幕模型 **并不能保证** 能够帮助 LLM 执行视频时刻定位任务

- Observation 5. Introducing additional query information into the description of video content can significantly improve the ability of LLMs to conduct video grounding, even with a small amount of additional information.

  - VQA-based 能够生成更好的（更受控制）视频描述
 
  - VQA-based method > Captions-based method

- Observation 6. The prompting method of instructing LLMs to separately judge the predictability and infer results can significantly improve the performance of video grounding.

  - Prompt 能够显著提升普通 LLM 模型的视频定位能力

- Observation 7. LLMs infer from the actually received information and complete the video grounding task,  rather than randomly guessing

  - LLM 确实在试图推断视频时刻定位的答案，而不是随机猜测

![](https://img.ithome.com/newsuploadfiles/2023/12/07704d4b-bc51-4bd0-a3e8-1b618cf2bc84.png?x-bce-process=image/format,f_avif)

- Observation 8. The reason for the failure case is mainly from the vague description of the visual models, and  the secondary one is the insufficient reasoning ability of LLMs in the case of weak information.

  - 失败的原因主要来自 1）视觉模型的 **模糊描述**，其次是 2）LLM 在 **信息较弱** 的情况下推理能力不足。

### 成功与失败示例

LLM4VG 的成功与失败案例如下图所示：

![成功与失败的示例](https://img.ithome.com/newsuploadfiles/2023/12/8f9719ce-8d5b-4561-aeea-b3c27ad90cdf.png?x-bce-process=image/format,f_avif)

其中，蓝色标注的表示 Positive 的文本，有利于 LLM 进行视频时刻定位；红色标注的表示 Negative 的文本，不利于 LLM 进行视频时刻定位。

由于 Video Description Generator 可能会生成与查询文本语义不相关的视频描述，从而导致 LLM 无法完成视频时刻定位，从而给出 "‘Based on the given caption, it is not possible to determine the grounding time for the query" 的回答。

> 这些失败的案例和成功的案例证明，LLM 确实在试图推断视频时刻定位的答案，而不是随机猜测。


## 参考：

- IT 之家：[清华大学研发 LLM4VG 基准：用于评估 LLM 视频时序定位性能](https://www.ithome.com/0/742/391.htm)

- 论文：[LLM4VG: Large Language Models Evaluation for Video Grounding](https://arxiv.org/abs/2312.14206)