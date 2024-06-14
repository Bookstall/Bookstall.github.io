---
layout: post
author: bookstall
tags: LLM, Vector DB
categories: [LLM, Vector DB]
excerpt: 介绍一下向量数据库的相关信息，并使用 LangChain 进行简单的实战
keywords: LLM, Vector DB
title: 向量数据库：LLM 的检索增强剂
mathjax: true
---

## 背景

在上一篇文章中，我们探讨了大语言模型的局限性，其中很重要的一点就是：**大语言模型对 tokens 数量的限制**，这个限制使得我们在开发大语言模型应用时面临很多顾虑。

![OpenAI ChatGPT-3.5](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7ed5671c1ae045ee94d11319e53ead43~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=620&h=239&s=51366&e=png&b=ffffff)

![OpenAI GPT-4](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6c8fc7613e6d4664a84063aab2201afe~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=612&h=346&s=81255&e=png&b=ffffff)

**向量数据库** 就是解决这个问题的方式之一。


## 1、向量数据库

### 1.1、基础概念

在数学中，向量是一个有序的数值序列。例如，二维平面中的一个点的位置可以用两个实数的向量 $$(x, y)$$ 来表示。同理，三维空间中的点可以用 $$(x, y, z)$$ 表示。 而在计算机科学中，这些点可以表示为事务的特征或属性，向量数据库就是用来存储这些点的特征或属性的。如下图所示：

![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d72db8c4c89e4bc99ee4ed12b14cd146~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=1419&h=736&s=928633&e=png&b=e3e3e3)

上面我们说了，向量数据库存储的这些 "点" 其实是事务的特征，那么具体是指什么呢？

假设我们是一个犬类动物爱好者，我们可以通过 **体型大小、毛发长度、鼻子长短** 等特征为狗狗分类，那么如果将犬类的体型大小、毛发长度用二维向量记录下来，就是下面这个样子：

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ce85b669f1974732a5c7d3b5bcb43134~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=1108&h=166&s=265543&e=png&b=e0dede)

其中，$$X$$ 轴表示犬类体型，取值范围从大到小为 0 到 1。

如果现在再加上毛发长度的 $$Y$$ 轴，就是下面的样子：

![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/423207acc82943b0a447c0cdf70f29c6~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=1166&h=509&s=654016&e=png&b=e3e2e2)

接下来，我们再延伸到三维坐标系，将犬类鼻子的长短记录到Z轴，便得到了下面的三维向量数据：

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b484078f612e46fe9bb7b64a1f64011b~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=1425&h=800&s=1322553&e=png&b=e0dfdf)

于是乎我们就得到了基于犬类体型大小、毛发长度等鼻子长短等特征的三维向量特征点，也就是 $$X、Y、Z$$ 轴的坐标，这些数据便是向量数据，存储到向量数据库。


向量数据库有了这些数据，便可以提供向我们的向量检索。例如，我们想养一只与哈士奇相似的狗，那么会推荐金毛（0.6，0.65，0.66），而不会推荐泰迪（0.1，0.45，0.23）。

---

当然了，我们不能只根据三个特征就推荐你养哪只狗狗，很明显这样不够精确。那么我们可以将犬类更多的特征，例如眼睛大小、服从性、攻击性等用向量数据记录下来。

也许你很难想象这些数据记录到四维、五维空间会是什么样子，但这不重要，我们只需要知道这些特征换成向量数据就是在向后面追加数字即可，例如：（0.53，0.4，0.75，0.11，0.23，……）

![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/bb7e6399fdb44a39bb72d17061bc3886~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=1008&h=776&s=871764&e=png&b=efeceb)

我们可以用这种方式表示所有事物，不管是具象的还是抽象的，例如，一段话、一张照片、喜怒哀乐、悲欢离合。而且数据的维度越高，描述的数据就越精确。

---

如下图所示，OpenAI 的文本向量模型 `text-embedding-ada-002` 可以输出 1536 维的数据。实际上在真实的生产环境中，上千的向量维度和上亿的向量数据都是正常的。

![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/566eb5be1ae64233839e9d6251c96ea4~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=709&h=161&s=21173&e=png&b=ffffff)


### 1.2、检索算法：ANN

ANN（相似最近邻搜索）算法是一种用于在大规模数据集中快速找到一个或多个近似最近邻的技术。ANN算法的类型很多，它们采用不同的技术实现和策略来在大规模数据集中加速最近邻搜索。

例如：Flat、k-means、LSH 等等。相似最近邻搜索问题是向量数据库产品的核心，一个成熟的向量数据库通常集成了多种 ANN 算法，并在搜索时综合各种指标选择最合适的策略来利用这些算法。

#### 1.2.1、Flat

关于搜索我们首先能想到的就是遍历搜索，遍历比较目标点，最终找个你想要的数据，这种方式就是 Flat 的实现逻辑，即：**暴力搜索、平推遍历**。

Flat 虽然很精确，但它的短板也很明显，那就是因为要遍历比较数据，它的 **检索速度非常慢**，在海量数据线我们一般不会使用它。

#### 1.2.2、k-means

聚类算法的原理是：将数据集中的样本划分为若干组，使得同一组内的样本彼此相似，而不同组之间的样本差异较大。聚类算法的种类有很多，k-means 是其中最常见的算法。

k-means 对数据的处理如下图所示，在搜索时我们只需要找出查询向量最近的哪个质心，然后在这个质心的簇中查找数据即可，达到缩小搜索范围的目的。

![k-means 聚类算法示意图](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c9224ac3e2264d80a7ab6cf1487aa864~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=1200&h=500&s=184020&e=png&b=fcf8f7)

就像 Flat 算法，聚类算法 k-means 也有自己的缺点。如下图所示：虽然查询向量 $$[0.74, 0.69]$$ 距离他最近的向量点在聚类中心 B 的簇内，但由于它距离聚类中心 A 更近，这种情况下就只能搜索到 A 簇的数据（k-means 会认为查询向量属于 A 类）。

![k-means 算法的缺点示意图](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0f4f87bb544443a9898ad8d2214a2df8~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=1671&h=888&s=1547907&e=png&b=e3e3e3)

解决上面的问题的方式很多，例如：**K-Means++ 初始化、谱聚类、层次聚类、DBSCAN** 等。

或者我们换一个算法——**位置敏感 Hash 算法（LSH）**来改善查询。

#### 1.2.3、LSH（Locality-sensitive Hashing）

LSH 使用一组哈希函数将相似向量映射到 “桶” 中，从而使相似向量具有相同的哈希值，这样就可以通过比较哈希值来判断向量之间的相似度。如下图所示：

![LSH 算法示意图](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e8c509daf57f4d18a54dfdbebcc71778~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=1400&h=787&s=516525&e=png&b=fefefe)

LSH 算法也有它的不足，例如在大型数据集中使用随机投影算法获取 hash 值时，生成质量好的随机的投影矩阵 **计算成本** 会很高。

### 1.3、ANN 的基准

[ANN 基准（ANN Benchmarks）](https://ann-benchmarks.com/) 是一种用于评估各种向量数据库和近似最近邻（ANN）算法性能的工具，主要包括以下指标：

- **数据集和参数规格**：ANN 基准提供大小、类型、维度不同数据集。每套数据集匹配一套参数，如：检索的最近邻数量、使用的距离计算公式等。

- **召回率（search recall）**：ANN 基准计算召回率，即返回的 k 个近邻中包含真正最近邻的比例。召回率是用于评估系统向量检索准确性的重要指标。

- **QPS**：ANN 基准还可以计算QPS（query per second），即向量数据库和 ANN 算法处理 query（查询请求）的速度。

  - QPS 是用于评估系统速度和可扩展性的重要指标。

我们可以使用 ANN 基准，在同一条件下比较不同向量数据库和 ANN 算法的性能，从而更快速找到最合适的选择。

> 算法并非解决问题的 "银弹"，每种算法都有自己的优势的短板，我们在使用时需要根据实际情况加以区分。 此外 ANN 算法也是向量数据库产品的核心，你可以用 ANN 基准来对向量数据库做评估，找到你需要的产品。


### 1.4、其他算法

此外还有很多算法，他们有各自的优点和短板，例如：

- HNSW（Hierarchical Navigable Small World）分层导航小世界算法

- k-平均演算法

- ……

## 2、向量数据库产品

随着这一轮人工智能的热潮，向量数据库的热度也从年初一直延续直径。目前市面上的向量数据库产品以及很丰富了。

![向量数据库在 Google 搜索中的热度指数](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d23b88cc90034fdbb62def0f5b8268d8~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=1194&h=709&s=156025&e=png&b=fefefe)

按照本地部署和云部署分类如下：

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c707927310f44c87be2572da7238e470~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=1401&h=937&s=160369&e=png&b=ffffff)

按照实现方式、开源程度可以做如下区分：

![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6d77530cf64a40ef8d42a04704242599~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=1400&h=832&s=316212&e=png&b=ffffff)

每种向量数据库支持的搜索算法也不尽相同：

![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7ae99304b6d9419da08cf9df5fadae1c~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=1186&h=712&s=123211&e=png&b=fefefe)

---

除此之外，在实际的业务场景中，向量数据库选型中，你还需要关注如下问题：

- **分布式能力**：CAP 如何取舍的，可用性和稳定性如何。

- **支持的数据类型和维度**：确保向量数据库支持你的数据类型和维度。有些数据库可能更适合特定类型的数据，如文本、图像或数值型数据。

- **可扩展性**： 确认向量数据库的可扩展性。考虑到项目可能的增长，确保数据库可以有效地处理大量的数据和查询请求。

- **API 和集成**： 评估数据库的 API 和集成能力。确保数据库可以轻松地集成到你的应用程序中，并且提供方便易用的 API。

- **安全性**： 安全性是任何数据库选择的一个关键因素。确保向量数据库提供适当的安全性特性，如数据加密、数据隔离、访问控制和身份验证。

- **社区和支持**： 考察数据库的社区支持和文档。一个活跃的社区通常能够提供更好的支持和解决问题的资源。

- **成本**： 评估数据库的总体成本，包括许可费、运维成本以及可能的扩展成本。确保数据库符合你的预算和资源限制。

---

如果你没有接触过向量数据库，建议了解下以下几款产品。

- 专业的向量数据库产品: chromadb、milvus、pinecone

- 具有向量数据库能力的产品：PostgreSQL & pgvector 、ElasticSearch 8.0 +


## 3、向量数据库增强 LLM

![向量数据库的交互逻辑](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b0caa2720e15487cbebefbb3a3297b4f~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=4096&h=1701&s=795072&e=png&b=effdf2)

这张图是向量数据库的交互逻辑，LangChain 可以对接不同的向量数据库产品，让向量数据库负责数据的检索，对接 LLM 模型，给出更加快速精确的答案。

以 **ChromeDB** 为例，演示了如何用 LangChain 对接向量数据，做一个本地的文档知识库。

在这篇文章里，我们在开发本地文档知识库时做了如下几件事情：

1.配置环境、Python、LangChain、ChatGLM2、 ChromaDB 等

2.将本地数据切片向量化：Docs -> Embeddings -> ChromaDB

3.编码对接 ChatGLM2、ChromaDB 等，完成开发


### 3.1、环境配置

```shell
pip install chromadb
```

### 3.2、分词（Tokenization）

> ChromeDB 支持 `doc`、`txt`、`pdf` 等格式的数据

将本地数据切片向量化，这是向量数据库与 LLM 对接的核心。

在使用向量数据库时，**分词（tokenization）**的意义与处理自然语言文本的相关任务密切相关。分词是将连续的文本数据切分成词（或称为单词）的过程。如下图所示。对于自然语言处理（NLP）和文本检索等任务，分词是一个重要的预处理步骤，因为它将文本数据转化为更易于处理的语言单元，每个语言单元可以是一个字母、数个词或一句话。

![Tokenizer 分词示意图](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/097608f506bf4a50807feda61bb4e07b~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=2294&h=800&s=189299&e=png&b=ffffff)

在 LangChain 中，我们可以通过 `tiktoken` 分词器对文档进行拆分，`tiktoken` 是由 OpenAI 开源的快速 BPE 分词器。

```python
# 根据 token 拆分文本
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, 
    chunk_overlap=0
)

texts = text_splitter.split_text(state_of_the_union)
```


### 3.3、Embeddings
**Embeddings（嵌入）**：向量数据库存储的是向量数据，我们的本地语料需要通过 Embeddings 的方式将自然语言转成成向量，也就是将数据映射到低维空间的表示形式。如下图所示：

![Embedding 示意图](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/dc6dce14f22f45248aebd98b26ac5bd4~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp#?w=426&h=265&s=18080&e=png&b=ffe6c9)

最常见的例子是 **词嵌入（Word Embeddings）**，它是将单词映射到实数向量的表示。**词嵌入模型**（如 Word2Vec、GloVe、FastText）通过学习大量文本语料库中的上下文关系，将每个单词映射到一个固定维度的实数向量。这样的词嵌入向量能够捕捉到单词之间的语义关系，使得在向量空间中相似的词语距离更近。

在 LangChain 中，我们可以使用 `OpenAIEmbeddings` 将文档转换成向量，并存储到向量数据库。

```python
# 初始化 openai 的 embeddings 对象
embeddings = OpenAIEmbeddings()

# 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息
# 并临时存入 Chroma 向量数据库，用于后续匹配查询
docsearch = Chroma.from_documents(split_docs, embeddings)
```

---

我们之前有提到文本向量可以通过 OpenAI 的 `text-embedding-ada-002` 模型生成，相似的：

- 图像向量可以通过 `clip-vit-base-patch32` 模型生成

- 音频向量可以通过 `wav2vec2-base-960h` 模型生成

- 还有一些对中文支持更好的，例如：`shibing624/text2vec-base-chinese`

这些向量都是通过 AI 模型生成的，所以它们都是具有语义信息的。


### 3.4、对接 LLM

最终我们通过 LangChain 的问答链实现与用户的交互：

```python
# 初始化 openai embeddings
embeddings = OpenAIEmbeddings()

# 将数据存入向量存储
vector_store = Chroma.from_documents(documents, embeddings)

# 通过向量存储初始化检索器
retriever = vector_store.as_retriever()

system_template = """
Use the following context to answer the user's question.
If you don't know the answer, say you don't, don't try to make it up. And answer in Chinese.
-----------
{question}
-----------
{chat_history}
"""

# 构建初始 messages 列表，这里可以理解为是 openai 传入的 messages 参数
messages = [
  SystemMessagePromptTemplate.from_template(system_template),
  HumanMessagePromptTemplate.from_template('{question}')
]

# 初始化 prompt 对象
prompt = ChatPromptTemplate.from_messages(messages)

# 初始化问答链
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1,max_tokens=2048),retriever,condense_question_prompt=prompt)

chat_history = []
while True:
  question = input('问题：')
  # 开始发送问题 chat_history 为必须参数,用于存储对话历史
  result = qa({'question': question, 'chat_history': chat_history})
  chat_history.append((question, result['answer']))
  print(result['answer'])
```



## 参考

- 掘金：
  
  - [向量数据库：高效检索与大语言模型的融合](https://juejin.cn/post/7318950135617585215)

  - [突破 LLM 的边界：解析 LLM 的局限与 LangChain 初探](https://juejin.cn/post/7317227758944436278)


