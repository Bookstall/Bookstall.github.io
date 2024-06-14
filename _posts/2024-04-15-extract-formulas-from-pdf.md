---
layout: post
author: bookstall
tags: LLM, PDF
categories: [LLM, PDF]
excerpt: 介绍了一些从 PDF 论文中提取数学公式的工具或方法，包括 Nought、grobid、LaTeX-OCR、Donut 以及 Mathpix Snip。
keywords: LLM, PDF
title: Nougat：从 PDF 论文中提取数学公式
mathjax: true
---

## 简介

> **Extracting formulas from scientific papers has always been a challenging task.**

本文介绍一些可以识别科学论文公式的工具，例如：

- [Nougat](https://arxiv.org/abs/2308.13418)：**N**eural **O**ptical **U**nderstandin**g** for **A**cademic Documen**t**s
  
  - 基于端到端、可训练的 Transformer 模型（包含 Encoder-Decoder），用于将 PDF 页面转换为标记（markup）

- [grobid](https://github.com/kermitt2/grobid)

  - 其性能表现不如 Nougat

- [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR/)

  - 其性能表现不如 Nougat

- [Donut](https://arxiv.org/abs/2111.15664)

  - Nougat 就是基于该模型结构

- [Mathpix Snip](https://mathpix.com/)：付费工具


## Nougat

Nougat 基于 Donut 的端到端结构，其具体结构如下所示：

![Nougat 的模型结构](https://pic4.zhimg.com/80/v2-a87fcf9585bfc66305fdc06a29016e7b_720w.webp)

其中，

- Swin Transformer Encoder：获取文档图像，并将其转化为特征向量（Embedding）

- Transformer Decoder：根据特征向量，通过自回归的方式将其转化为 Token 序列

Nougat 在 arXiv 测试集上的实验结果如下所示：

![Nougat's comparable result](https://pic4.zhimg.com/80/v2-0a9b828938258939072ad6526c2412ff_720w.webp)


### Nougat 安装

通过 pypi 进行安装：

```shell
pip install nougat-ocr
```

通过 github 进行安装：

```shell
pip install git+https://github.com/facebookresearch/nougat
```

### CLI（命令行）运行

这里以 [《Attention is all you need》](https://arxiv.org/pdf/1706.03762.pdf) 论文的第 5 页为例：

![论文的原始页面](/images/posts/Nougat/naugat-transformer-origin.png)

使用 `naugat-base` 模型进行识别与转换：

```shell
nougat ./1706.03762.pdf --model 0.1.0-base --page '5' -o ./results
```

运行结果如下所示：

```shell
WARNING:root:No GPU found. Conversion on CPU is very slow.
downloading nougat checkpoint version 0.1.0-base to path /root/.cache/torch/hub/nougat-0.1.0-base
config.json: 100% 560/560 [00:00<00:00, 1.78Mb/s]
pytorch_model.bin: 100% 1.31G/1.31G [00:06<00:00, 213Mb/s]
special_tokens_map.json: 100% 96.0/96.0 [00:00<00:00, 337kb/s]
tokenizer.json: 100% 2.04M/2.04M [00:00<00:00, 28.5Mb/s]
tokenizer_config.json: 100% 106/106 [00:00<00:00, 423kb/s]
/usr/local/lib/python3.10/dist-packages/torch/functional.py:507: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3549.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
  0% 0/1 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/nougat/model.py:437: UserWarning: var(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel). (Triggered internally at ../aten/src/ATen/native/ReduceOps.cpp:1760.)
  return torch.var(self.values, 1) / self.values.shape[1]
[nltk_data] Downloading package words to /root/nltk_data...
[nltk_data]   Unzipping corpora/words.zip.
INFO:root:Processing file 1706.03762.pdf with 1 pages
100% 1/1 [07:26<00:00, 446.25s/it]
```

Nougat 默认输出的 [MMD（Mathpix Markdown）](https://github.com/Mathpix/mathpix-markdown-it) 文件，与普通的 Markdown 文件不太兼容。可以通过 `--markdown` 来向下兼容 Markdown 文件。

---

直接使用 Typora 无法正确渲染相应的公式，因此，使用 VS Code 下载相应的插件 Mathpix Markdown，渲染后的结果如下所示：

![Naugat 转化结果](/images/posts/Nougat/naugat-transformer-result.png)

可以看到，Naugat 能够比较准确地对论文进行识别和转换，包括文本、公式。<u>但是，标题中标号 `3.3` 和 `3.4` 没有被成功转换。</u>

---

Nougat 完整的运行参数如下所示：

```shell
usage: nougat [-h] [--batchsize BATCHSIZE] [--checkpoint CHECKPOINT] [--model MODEL] [--out OUT]
              [--recompute] [--markdown] [--no-skipping] pdf [pdf ...]

positional arguments:
  pdf                   PDF(s) to process.

options:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE, -b BATCHSIZE
                        Batch size to use.
  --checkpoint CHECKPOINT, -c CHECKPOINT
                        Path to checkpoint directory.
  --model MODEL_TAG, -m MODEL_TAG
                        Model tag to use.
  --out OUT, -o OUT     Output directory.
  --recompute           Recompute already computed PDF, discarding previous predictions.
  --full-precision      Use float32 instead of bfloat16. Can speed up CPU conversion for some setups.
  --no-markdown         Do not add postprocessing step for markdown compatibility.
  --markdown            Add postprocessing step for markdown compatibility (default).
  --no-skipping         Don't apply failure detection heuristic.
  --pages PAGES, -p PAGES
                        Provide page numbers like '1-4,7' for pages 1 through 4 and page 7. Only works for single PDFs.
```


## 小结

> 上面的运行代码可以通过 [Google Colab Notebook](https://colab.research.google.com/drive/1-aOLRWA1WbewsOTolECne6tUnkWswZW3?usp=sharing) 获取

总的来说，Nougat 是一款出色的公式提取工具。

然而，作为一款端到端工具（它不需要任何与 OCR 相关的输入或模块，网络会隐式识别文本），它缺乏中间结果，而且定制选项似乎也很有限。

此外，Nougat 利用自回归前向传递来生成文本，这导致生成速度相对较慢，并增加了出现幻觉和重复的可能性。



## 参考

- [Unveiling PDF Parsing: How to extract formulas from scientific pdf papers](https://medium.com/@florian_algo/unveiling-pdf-parsing-how-to-extract-formulas-from-scientific-pdf-papers-a8f126f3511d)

  - 知乎：[揭开 PDF 解析的神秘面纱： 如何从 PDF 科学论文中提取公式](https://zhuanlan.zhihu.com/p/682975999)

- Nougat：

  - [Nougat's github](https://github.com/facebookresearch/nougat)

  - [Nougat's website](https://facebookresearch.github.io/nougat/)

  - [HuggingFace: nougat-small](https://huggingface.co/facebook/nougat-small)

  - [HuggingFace: nougat-base](https://huggingface.co/facebook/nougat-base)

- [mathpix-markdown-it](https://github.com/Mathpix/mathpix-markdown-it)

