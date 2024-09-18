---
layout: post
author: bookstall
tags: LLM
categories: [LLM]
excerpt: 当训练 or 微调一个 LLM 时，我们通常需要自己训练一个 Tokenizer。
keywords: LLM
title: 训练一个 Tokenizer
mathjax: true
---


## 1、背景



### 1.1、Tokenizer 的作用


### 1.2、Tokenizer 的种类



## 2、HuggingFace Tokenizer 的处理流程

Tokenizer 包括几个步骤：

- 规范化（任何认为必要的文本清理，例如删除空格或重音符号、Unicode 规范化等）

- 预标记化（将输入拆分为单词）

- 通过模型处理输入（使用预先拆分的词来生成一系列标记）

- 后处理（添加标记器的特殊标记，生成注意力掩码和标记类型 ID）

Tokenizer 的处理流程如图所示：

![Tokenizer的处理流程示意图](/images/posts/Tokenizer/tokenizer_processing.png)

> 图片来源：https://jinhanlei.github.io/posts/Transformers%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8-%E4%BA%8C-%E7%94%A8Tokenizer%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E8%AE%AD%E7%BB%83%E8%AF%8D%E8%A1%A8/

![HuggingFace Tokenizer的处理流程示意图](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/tokenization_pipeline-dark.svg)

> 图片来源：https://huggingface.co/learn/nlp-course/zh-CN/chapter6/4

我们通常把训练集叫做 “语料库”（Corpus），根据语料库得到一个词表，词表需要涵盖语料库中的所有词。下面详解这一词表的构建流程。

### 2.1、Normalizers（规范化）

### 2.2、Pre-tokenizers（预分词）

这一步进行预分词。根据切分粒度的不同，天然可以把英文单词拆成：字符、词，但这样有很大的弊端。

#### 2.2.1、按字符切分（Character-based）

把文本切分为字符，这样就只会产生一个非常小的词表，比如英文就只有 26 个字母和标点等，很少会出现词表外的 tokens。例如对 `Attention is all we need!` 按字符切分为：

```shell
A|t|t|e|n|t|i|o|n|i|s|a|l|l|w|e|n|e|e|d|!
```

但这样显然不太合理，因为字符本身并没有太多上下文的规律。更重要的是，这样仅仅一个单词就需要一堆向量去表示，遇到长文本时就爆炸了。

> 划分的 token 太长，计算 attention 时消耗太多时间

#### 2.2.2、按词切分（Word-based）

按词切分几乎是最直观的方法。

```shell
Attention|is|all|we|need|!
```

这种策略也同样存在问题。

- 对于中文，因为字之间没有空格天然分开成词，分词本身就是一项挑战。

- 对于英文，会将文本中所有出现过的独立片段都作为不同的 token，从而产生巨大的词表，而实际上词表中很多词是相关的，例如 “dog” 和 “dogs”、“run” 和 “running”，如果视作不同的词，就无法表示出这种关联性。

并且，这样会出现 **OOV（Out of vocabulary）**问题，在预测时有可能遇到语料库中从没出现过的词，分词器会使用一个专门的 `[UNK]` token 来表示，如果训练集不够大，词表里就那么几个词，那么用的时候，句子中会包含大量 `[UNK]`，导致大量信息丧失。因此，一个好的分词策略，应该尽可能不出现 `[UNK]`。



#### 2.2.3、按子词切分（Subword-based）

现在，广泛采用的是一种同时结合了按词切分和按字符切分的方式——按子词切分 (Subword tokenization)。

BERT、GPT 都采用这种做法，**高频词直接保留，低频词被切分为子词**。

Subword 算是一种对字符和词折中的办法。不仅子词之间有规律可循、单词不会切的过分长，而且只用一个较小的词表就可以覆盖绝大部分的文本，基本不会产生 `[UNK]`。

在统计词频前，需要预先切割词，切完才能去统计，Pre-tokenizers 就是切割的作用。英文本身就可以根据空格分割，但是对于中日韩这样的连续字符，如果按单字切完，词表将会巨大，几乎需要 130,000+ 个 Unicode 字符，更别说还要继续组词了！

---

于是，GPT-2 等采用了 ByteLevel 的算法，将中日韩等字符映射到 **256 个字符**。具体可以参考 [fairseq](https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/encoders/byte_utils.py) 和 [icefall](https://github.com/k2-fsa/icefall/blob/master/icefall/byte_utils.py)。将 [fairseq](https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/encoders/byte_utils.py) 里面的代码 copy 下来，加上：

```python
enc = byte_encode("唱、跳、rap、篮球")
print(enc)
print(byte_decode(enc))
print(len(BYTE_TO_BCHAR))
```

`唱、跳、rap、篮球` 被编码为 `åƔ±ãƀƁè·³ãƀƁrapãƀƁç¯®çƐƃ`，这啥？我也看不懂，但是找到 `rap` 在里面了吗？英文是不变的，每个中文字符会被映射成三个 `å` 这种玩意儿，而这种玩意儿加上空格、英文字母和标点等，一共只有 256 个，他们的排列组合可以构成几乎所有字符。

之后去统计这些玩意儿的共现频率就可，但考虑到可能会有一些无效的组合，比如中间少个 `Ɣ`，`byte_decode` 就出不来了，于是引入基于动态规划的最佳解码方法 `smart_byte_decode`：

```python
wrong_byte = "å±ãƀƁè·³ãƀƁrapãƀƁç¯®çƐƃ"
print(byte_decode(wrong_byte))
print(smart_byte_decode(wrong_byte))
```

将每个汉字用 [空格分隔](https://github.com/k2-fsa/icefall/blob/dca21c2a17b6e62f687e49398517cb57f62203b0/icefall/utils.py#L1370)，就可以按英文那般分词。做完分词，就可以来统计词频了，训练一个词表出来了。

### 2.3、Tokenizer-Models（构建词表的统计模型）

Tokenizer 的 Models 是最核心的部分，指构建词表的统计模型，有三大子词标记化算法：BPE、WordPiece 和 Unigram。

在下面的部分中，我们将深入研究三种主要的子词标记化算法：**BPE**（由 GPT-2 和其他人使用）、**WordPiece**（例如由 BERT 使用）和 **Unigram**（由 T5 和其他人使用）。在我们开始之前，这里是它们各自工作原理的快速概述。如果您还没有理解，请在阅读下一节后立即回到此表。


| Model         | BPE                                                            | WordPiece                                                                                               | Unigram                                                                       |
| ------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Training      | 从少量词汇开始，学习合并 token 的规则                          | 从少量词汇开始，学习合并 token 的规则                                                                   | 从大量词汇开始，学习删除 token 的规则                                         |
| Training step | 合并最常见的相邻 token 对                                      | 根据对的频率合并与具有最佳分数的对相对应的令牌，使每个单独 token 的频率较低（频率低的做为单独的 token） | 删除词汇表中的所有 token，这些 token 将最大程度地减少在整个语料库上计算的损失 |
| Learns        | 合并规则和词汇表                                               | 只是一个词汇                                                                                            | 包含每个 token 分数的词汇表                                                   |
| Encoding      | 将单词拆分为多个字符（characters），并使用在训练期间学到的合并 | 查找从词汇表中的开头开始的最长的子字（subword），然后对单词的其余部分执行相同的操作                     | 使用训练期间学到的分数找到最有可能被拆分的 token                              |

#### 2.3.1、Byte Pair Encoding（BPE）

> 参考：
>
> - HuggingFace：https://huggingface.co/learn/nlp-course/zh-CN/chapter6/5?fw=pt

字节对编码（BPE）最初被开发为一种压缩文本的算法,然后在预训练 GPT 模型时被 OpenAI 用于标记化。许多 Transformer 模型都使用它,包括 GPT、GPT-2、RoBERTa、BART 和 DeBERTa。

[BPE](https://arxiv.org/abs/1508.07909) 简单有效，是目前最流行的方法之一，GPT-2 和 RoBERTa 使用的 Subword 算法都是 BPE。BPE 的流程如下：

- 根据分词结果统计词频，得到{`词 </w>`: 词频}，词后加上末尾符是为了区分 “**est**imate” 和 “high**est**” 这类词；

- 统计字符的个数（比如那 256 个玩意），得到 {“字符”: 字符频} 表；

- 拿字符频最高的字符与下一字符合并，统计合并后的 Subword 频率；

- 将 Subword 频添加到{“字符”: 字符频}表，这时词表会扩大；

- 继续拿表中频率最高的去合并，到末尾符时停止这个词的合并；

- 重复直到预设的词表大小或最高频数为 1。

更详尽的推导可以参考 [这篇](https://zhuanlan.zhihu.com/p/424631681)。


#### 2.3.2、WordPiece

> 参考：
>
> - HuggingFace：https://huggingface.co/learn/nlp-course/zh-CN/chapter6/6?fw=pt

WordPiece 是 Google 为预训练 BERT 而开发的标记化算法。此后,它在不少基于 BERT 的 Transformer 模型中得到重用,例如 DistilBERT、MobileBERT、Funnel Transformers 和 MPNET。它在训练方面与 BPE 非常相似,但实际标记化的方式不同。

WordPiece 主要在 BERT 类模型中使用。与 BPE 选择 **频数最高的相邻子词合并** 不同的是，WordPiece 选择 **能够提升语言模型概率最大的相邻子词** 加入词表。

#### 2.3.3、Unigram

> 参考：
>
> - HuggingFace：https://huggingface.co/learn/nlp-course/zh-CN/chapter6/7?fw=pt

在 SentencePiece 中经常使用 Unigram 算法,该算法是 AlBERT、T5、mBART、Big Bird 和 XLNet 等模型使用的标记化算法。


[Unigram](https://arxiv.org/pdf/1804.10959.pdf) 的操作是和前两者反向的。不同于拼词，Unigram 是割词，首先初始一个大词表，接着通过概率模型不断拆出子词，直到限定词汇量。

可以从 WordPiece 的公式去理解。由于刚开始都是长词，词表是巨大的，通过拆概率小的词，保留概率大的词，从而缩小词表。

根据这些方法，可以根据自己的语料，训练一个垂直领域的词表，这些方法能够很好地将高频词、术语等统计出来。

### 2.4、Post-Processors（后处理）

在训练词表后，还可能需要对句子进行后处理。例如一些模型当我们分完词，还想给句子加入特殊的标记，例如 BERT 会给句子加入分类向量和分隔符 `[CLS] My horse is amazing [SEP]`，这时就需要Post-Processors。

## 3、继承一个已有的 Tokenizer

> 参考：
>
> - HuggingFace：https://huggingface.co/learn/nlp-course/zh-CN/chapter6/2?fw=pt

### 3.1、准备语料库



### 3.2、训练新的 Tokenizer



### 3.3、保存 Tokenizer

## 4、从头开始训练 HuggingFace Tokenizer

> 参考：https://huggingface.co/learn/nlp-course/zh-CN/chapter6/8?fw=pt

### 4.0、构建语料库

使用 [WikiText-2](https://huggingface.co/datasets/wikitext) 数据集

```python
from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]
```

`get_training_corpus()` 函数是一个生成器，每次调用的时候将产生 1,000 个文本，我们将用它来训练标记器。

Tokenizers 也可以直接在文本文件上进行训练。以下是我们如何生成一个文本文件，其中包含我们可以在本地使用的来自 WikiText-2 的所有文本/输入：

```python
with open("wikitext-2.txt", "w", encoding="utf-8") as f:
    for i in range(len(dataset)):
        f.write(dataset[i]["text"] + "\n")
```

### 4.1、WordPiece Tokenizer


### 4.2、BPE Tokenizer

现在让我们构建一个 GPT-2 标记器。与 BERT 标记器一样，我们首先使用 Tokenizer 初始化一个BPE 模型：

```python
tokenizer = Tokenizer(models.BPE())
```

和 BERT 一样，如果我们有一个词汇表，我们可以用一个词汇表来初始化这个模型（在这种情况下，我们需要传递 vocab 和 merges），但是由于我们将从头开始训练，所以我们不需要这样去做。 我们也不需要指定 “unk_token”，因为 GPT-2 使用的字节级 BPE，不需要 “unk_token”。

---

GPT-2 不使用归一化器，因此我们跳过该步骤并直接进入预标记化：

```python
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

我们在此处添加到 ByteLevel 的选项 `add_prefix_space=False` 是不在句子开头添加空格（默认为 True）。 我们可以看一下使用这个标记器对之前示例文本的预标记：

```python
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!")
```

```shell
[('Let', (0, 3)), ("'s", (3, 5)), ('Ġtest', (5, 10)), ('Ġpre', (10, 14)), ('-', (14, 15)),
 ('tokenization', (15, 27)), ('!', (27, 28))]
```

---

接下来是需要训练的模型。对于 GPT-2，唯一的特殊标记是文本结束标记：

```python
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
```

与 WordPieceTrainer 以及 `vocab_size` 和 `special_tokens` 一样，我们可以指定 `min_frequency` 如果我们愿意，或者如果我们有一个词尾后缀（如 `</w>`)，我们可以使用 `end_of_word_suffix` 设置它。

这个标记器也可以在文本文件上训练：

```python
tokenizer.model = models.BPE()
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
```

让我们看一下示例文本的标记化后的结果：

```python
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
```

```shell
['L', 'et', "'", 's', 'Ġtest', 'Ġthis', 'Ġto', 'ken', 'izer', '.']
```

---

我们对 GPT-2 标记器添加字节级后处理，如下所示：

```python
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
```
`trim_offsets = False` 选项指示我们应该保留以 ‘Ġ’ 开头的标记的偏移量：这样偏移量的开头将指向单词之前的空格，而不是第一个单词的字符（因为空格在技术上是标记的一部分）。 让我们看看我们刚刚编码的文本的结果，其中 'Ġtest' 是索引第 4 处的标记：

```python
sentence = "Let's test this tokenizer."
encoding = tokenizer.encode(sentence)
start, end = encoding.offsets[4]
sentence[start:end]
```

```shell
' test'
```

---

最后，我们添加一个字节级解码器：

```python
tokenizer.decoder = decoders.ByteLevel()
```

我们可以仔细检查它是否正常工作：

```python
tokenizer.decode(encoding.ids)
```

```shell
"Let's test this tokenizer."
```

---

现在我们完成了，我们可以像以前一样保存标记器，并将它包装在一个 PreTrainedTokenizerFast 或者 GPT2TokenizerFast 如果我们想在 🤗 Transformers中使用它：

```python
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
)
```

或者：

```python
from transformers import GPT2TokenizerFast

wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
```

### 4.3、Unigram Tokenizer



## 5、从零开始训练 BPE Tokenizer



## 6、关于 Tokenizer 的思考




## 更多

- tiktokenizer：https://tiktokenizer.vercel.app

- tiktoken from OpenAI: https://github.com/openai/tiktoken

- sentencepiece from Google https://github.com/google/sentencepiece


## 参考

- bilibili：

  - 

- 知乎：
  
  - 

- HuggingFace：[NLP Course](https://huggingface.co/learn/nlp-course/zh-CN/chapter6/1?fw=pt)

- Youtube：[Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)

  - [Google colab for the video](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)

  - GitHub repo for the video: [minBPE](https://github.com/karpathy/minbpe)



