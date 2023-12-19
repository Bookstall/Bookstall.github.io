---
layout: fragment
title: HuggingFace 镜像
tags: [HuggingFace]
excerpt: HuggingFace 镜像
keywords: HuggingFace
mathjax: true
---

## 方式一：modelee

[modelee](https://gitee.com/modelee) 当前主要是提供 huggingface 模型和数据集的镜像，方便国内开发者快速获取。

- 本站供您下载模型镜像文件（以下称 “镜像文件”）的知识产权归属于该镜像文件的开发者所有。您的下载和使用镜像文件的行为应遵守该等知识产权的约定。若违侵犯该等知识产权，在法律规定的范围内，由此带来的责任由您承担。

- 除第 1 条的镜像文件外，本站的其他内容，包括图片、网站架构与画面的安排、网页设计、文字、图表、代码、SDK、API、LOGO 等知识产权及其他合法权益，包括商标权、著作权与专利权等知识产权，均归我们或我们的关联方所有。未经我们或关联公司的事先书面许可，任何人不得以包括通过机器人、蜘蛛等程序或设备监视、复制、传播、展示、镜像、上载、下载等方式擅自使用本站内除第 1 条镜像文件之外的所有内容。


## 方式二：hf-mirror

[hf-mirror](https://hf-mirror.com/) 是一个 HuggingFace 镜像站，提供 HuggingFace 模型和数据集的搜索。

使用 url 下载模型或者数据集时，将 `huggingface.co` 直接替换为 `hf-mirror.com`。使用浏览器或者 `wget -c`、`curl -L`、`aria2c` 等命令行方式即可。

下载需登录的模型需命令行添加 `--header hf_***` 参数，添加 HuggingFace 的 `token`。


## 参考

- 知乎问题：[有没有下载 Hugging Face 模型的国内站点？](https://www.zhihu.com/question/371644077)
