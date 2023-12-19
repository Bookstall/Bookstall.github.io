---
layout: post
author: bookstall
tags: AI, Diffusion
categories: [AI, Diffusion]
excerpt: ControlNet：为 Stable Diffusion 插上翅膀
keywords: AI, Diffusion
title: ControlNet：为 Stable Diffusion 插上翅膀
mathjax: true
---


## 前言

从骑马的宇航员到三次元小姐姐，在不到一年的时间里，**AI 绘画** 似乎已经取得了革命性的进展。

### DALL·E 2

这个「骑马的宇航员」由 OpenAI 2022 年 4 月推出的文生图模型 **DALL・E 2** 绘制。它的前辈 ——DALL・E 在 2021 年向人们展示了直接用文本生成图像的能力，打破了自然语言与视觉的次元壁。

<img src="https://image.jiqizhixin.com/uploads/editor/af95d822-5ed6-4adc-a7a6-4eaf1356c61f/640.png" alt="DALL·E 2 生成的图片" style="zoom:80%;" />

在此基础上，DALL・E 2 更进一步，允许人们对原始图像进行 **编辑**，比如在画面中添加一只柯基。这一个看似简单的操作其实体现了 **AI 绘画模型可控性** 的提升。

<img src="https://image.jiqizhixin.com/uploads/editor/c316fc76-b3f7-4aac-ac44-450783dd1027/640.jpeg" alt="DALL・E 2 对图片进行编辑" style="zoom:80%;" />

### Stable Diffusion

不过，就影响力而言，2022 年最火的文生图模型并不是 DALL・E 2，而是另一个和它功能相似的模型——**Stable Diffusion**。

和 DALL・E 2 一样，**Stable Diffusion** 也允许创作者对生成的图像进行编辑，但优势在于，这个模型是开源的，而且可以在消费级 GPU 上运行。因此，在 2022 年 8 月发布之后，Stable Diffusion 迅速走红，短短几个月就成了最火的文生图模型。

<img src="https://image.jiqizhixin.com/uploads/editor/44116ae9-a6e5-40e9-8b4c-48df6b75b2ec/640.gif" alt="Stable Diffusion 生成过程示例" style="zoom: 80%;" />

在此期间，人们也在进一步探索各种控制这类模型的方法，比如 Stable Diffusion 背后团队之一的 Runway 公司发布了一个[图像擦除和替换（Erase and Replace）工具](http://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650859439&idx=1&sn=d903e2d4c2ec062cffcd9d56dbcb2e61&chksm=84e523d1b392aac73c6399e43516cd23f5633ecdca7c3b55aa87e5bd6e9a0eff1d43e03f7edf&scene=21#wechat_redirect)，该工具可以修改图像任何部分。用户需要做的就是擦除该区域并编写自然语言描述，剩下的交给程序就可以了。

<img src="https://image.jiqizhixin.com/uploads/editor/4a24eaea-c88b-4c23-8ed4-bd268c152565/640.gif" alt="Erase and Replace 工具" style="zoom:80%;" />

谷歌和波士顿大学的研究者则提出了一种「个性化」的文本到图像扩散模型 [DreamBooth](http://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650860881&idx=2&sn=94525787e454e31537c7d9a537241857&chksm=84e5292fb392a039787c98aba3df5c4d3538d343186f7c0971699bed0b2261d7231571cbbec4&scene=21#wechat_redirect)，用户只需提供 3~5 个样本 + 一句话，AI 就能定制照片级图像。

<img src="https://image.jiqizhixin.com/uploads/editor/c4570556-1638-41b8-bd27-3e4a6a502c32/640.png" alt="DreamBooth" style="zoom:80%;" />

此外，来自 UC 伯克利的研究团队还提出了一种根据人类指令编辑图像的新方法 [InstructPix2Pix](http://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650861567&idx=1&sn=dd50f5c9532e5e4b8f13c914742a6202&chksm=84e52b81b392a297b1581a6b7f5734e179bc315d816dfa6f84dccfa26f47db66729eaf0ae5e9&scene=21#wechat_redirect)，这个模型结合了 GPT-3 和 Stable Diffusion。给定输入图像和告诉模型要做什么的文本描述，模型就能遵循描述指令来编辑图像。例如，要把画中的向日葵换成玫瑰，你只需要直接对模型说「把向日葵换成玫瑰」。

<img src="https://image.jiqizhixin.com/uploads/editor/5cd10e97-1223-4db2-a9f4-fb9b0bc2a06f/640.png" alt="InstructPix2Pix" style="zoom:80%;" />



## ControlNet

进入 2023 年，一个名为 **ControlNet** 的模型将这类控制的灵活度推向了高峰。

### 效果







### 原理



<img src="https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW8ywqBEJZr5oL2yF3FyptaR1x9HMz5M4bSKg9YxdAwpcicGkNrXNdOyCcJesldgMIEyJvLXUHpvdcw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" style="zoom: 50%;" />





<img src="https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW8ywqBEJZr5oL2yF3FyptaRABTiaZpvZbCTcpCZiarRjrXP8jsnGvgRLEwUicuJ12gLciaBCSRKt8ef9Q/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="ControlNet" style="zoom: 50%;" />



Stable Diffusion 本质上是一个 U-Net，可以将 ControlNet 加到 Stable Diffusion 中，如下所示。



<img src="https://github.com/lllyasviel/ControlNet/raw/main/github_page/sd.png" alt="Stable Diffusion + ControlNet" style="zoom:67%;" />





### 应用

利用 ControlNet 和 EbSynth 等工具重新进行 **室内装潢设计**。

> 来源：https://creativetechnologydigest.substack.com/p/controlling-artistic-chaos-with-controlnet

<img src="https://image.jiqizhixin.com/uploads/editor/07ef2254-91c5-419f-9168-3f625bd00c6b/640.gif" alt="室内装潢设计" style="zoom: 67%;" />



利用 ControlNet 和 Houdini 工具生成 3D 模型。

<img src="https://image.jiqizhixin.com/uploads/editor/60ac9801-669e-4719-9695-13bd674496ca/640.gif" alt="生成 3D 模型" style="zoom: 80%;" />



用 Dreambooth 和 ControlNet **改变 2D 图像光照**，可用于照片、视频的后期制作。

> 来源：*https://www.reddit.com/r/StableDiffusion/comments/1175id9/when_i_say_mindblowing_i_mean_it_new_experiments/*

<img src="https://image.jiqizhixin.com/uploads/editor/c638e24a-4941-4a36-8997-8236a932ceb6/640.gif" alt="改变 2D 图像光照" style="zoom: 80%;" />



用 ControlNet 和 EbSynth 实现 **动画转真人**。虽然效果还不太好，但已经显示出了把动漫改编成真人版但无需演员出镜的潜力。

> 来源：*https://www.reddit.com/r/StableDiffusion/comments/117ewr9/anime_to_live_action_with_controlnet_ebsynth_not/*

<img src="https://image.jiqizhixin.com/uploads/editor/d5f123d7-91b2-430b-931a-01048b1b9a4a/640.gif" alt="动画转真人" style="zoom:80%;" />





某设计师利用 ControlNet 生成的著名品牌。

> 来源：https://twitter.com/fofrAI/status/1628882166900744194

<img src="https://image.jiqizhixin.com/uploads/editor/98bfef1e-f05a-4114-9385-3d36f3e2698b/640.png" style="zoom: 33%;" />





## 参考

- ControlNet Repo：https://github.com/lllyasviel/ControlNet

- 论文：[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)

- 机器之心：[ControlNet star量破万！2023年，AI绘画杀疯了？](https://www.jiqizhixin.com/articles/2023-03-02-10)

- 机器之心：[AI降维打击人类画家，文生图引入ControlNet，深度、边缘信息全能复用](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650868980&idx=4&sn=369ba4f0d80b6fe8a5b92ed2be1e53e0&chksm=84e4c88ab393419cacb3a82ef07e2faafe6d534ba25fde26317695e98a3790c9fcf1df03abba&scene=21#wechat_redirect)





