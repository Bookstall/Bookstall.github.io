---
layout: post
author: bookstall
tags: AI
categories: [AI]
description: Video Retrival Task's Papers
keywords: AI
title: 视频检索任务论文整理
mathjax: true
---

视频检索
视频片段检索：给定查询内容（通常是文本/自然语言、也可以是视频片段、图片、音频片段等）和一个未裁剪的长视频，视频片段检索任务需要在这个长视频中检索出一个片段，使得这一片段的语义信息能够与给定的查询内容相匹配/对应。

golden moment

与之相关的研究：

- 深度学习

- 多模态数据的表示

- 目标检测

- 文本图像匹配

- 时序动作检测：在一段未剪辑的视频中找到动作开始和结束的时间，并对动作进行分类





## Text-Video Retrival

基于文本/自然语言查询的视频片段检索任务



相关术语：

- **时域语言定位（Temporally Language Grounding）**
  - 给定一句自然语言描述，模型需要在未剪裁的视频（Untrimmed Videos）中确定该描述的活动发生的时间片段（起始时间，终止时间）
- **时序动作定位 (Temporal Action Localization)** 也称为时序动作检测 (Temporal Action Detection)，是视频理解的另一个重要领域
  - 时序动作定位不仅要预测视频中包含了什么动作，还要预测动作的起始和终止时刻
  - 相比于动作识别，时序动作定位更接近现实场景

- Video Grounding
- text-to-clip retrieval
- query-based moment retrieval

- Temporal Sentence Grounding
- Natural Language Moment Retrieval

- temporal moment localization





数据集

- TACoS
  - 项目地址：https://www.coli.uni-saarland.de/projects/smile/page.php?id=software

- Charades-STA
- DiDeMo
  - paperwithcode：[Natural Language Moment Retrieval on DiDeMo](https://paperswithcode.com/sota/natural-language-moment-retrieval-on-didemo) ==VLG-Net==
  - github：https://github.com/LisaAnne/LocalizingMoments

- ActivityNet Captions
  - 项目地址：https://cs.stanford.edu/people/ranjaykrishna/densevid/

![Video Retrival 常用数据集](https://img-blog.csdnimg.cn/591ecd3d8417430a8e8902e94b18d117.png)





评估指标（metric）

`R@n,IoU=m`

排名前 n 个模型输出结果中，至少存在一个的 IoU 值超过 ground truth m 的比例



![](https://pic2.zhimg.com/80/v2-1ca5c6d2468606ac844bfbe1b15245b5_720w.webp)



### 2017（2/2）

> [!Info+年度关键词]
>
> **两条路**

| 标题                                                        | 方法名                                                 | 会议      | 数据集                         | 备注     |
| ----------------------------------------------------------- | ------------------------------------------------------ | --------- | ------------------------------ | -------- |
| TALL: Temporal Activity Localization via Language Query :o: | CTRL 模型（Cross-modal Temporal Regression Localizer） | ICCV 2017 | TaCoS<br />Charades-STA :star: | 开山之作 |
| Localizing Moments in Video with Natural Language :o:       | MCN 模型（Moment Context Network）                     | ICCV 2017 | DiDeMo :star:                  |          |



#### TALL-CTRL :white_check_mark:

> 《TALL: Temporal Activity Localization via Language Query》
>
> - URL：http://arxiv.org/abs/1705.02101
> - Code：https://github.com/jiyanggao/TALL ==TensorFlow==
> - 会议：ICCV 2017

开山之作，点出该问题需要解决的两大困难：

- 作为跨模态任务，如何得到适当的文本和视频表示特征，以允许跨模态匹配操作和完成语言查询。
- 理论上可以生成无限粒度的视频片段，然后逐一比较。但时间消耗过大，那么如何能够从有限粒度的滑动窗口做到准确的具体帧定位。

方法：

- 特征抽取：使用全局的 sentence2vec 和 C3D
- 模态交互：逐元素加、逐元素乘、FC、拼接
- 多任务：对齐分数 + 回归（Temporal Regression）

![CTRL 模型结构](https://pic1.zhimg.com/v2-0ded7348b913b2c9978031c7b90d29c0_r.jpg)





#### MCN :white_check_mark:

> 《Localizing Moments in Video with Natural Language》
>
> - URL：https://arxiv.org/abs/1708.01641
> - Code：https://github.com/LisaAnne/LocalizingMoments ==Official, Caffe2==
> - DiDeMo 数据集：[Google Driver](https://drive.google.com/drive/u/0/folders/1heYHAOJX0mdeLH95jxdfxry6RC_KMVyZ) & [Google Driver](https://drive.google.com/drive/u/0/folders/1_oyJ5rQiZboipbMl6tkhY8v0s9zDkvJc)
> - 会议：ICCV 2017

方法：

- 特征抽取：
  - 使用全局的 LSTM；
  - 使用全局与局部的 VGG；
  - 提取 Temporal Endpoint Feature；
- 模态交互：无
- 多任务：模态间的均方差（而不是预测对齐分数），同时使用 RGB 和 optical flow（光流）

![MCN 模型结构](https://pic1.zhimg.com/80/v2-5d8594efc155281e22687dc3e59c7de8_720w.webp)



### 2018（2/5）

> [!Info+年度关键词]
>
> **上下文（Context）**

| 标题                                                         | 方法名 | 会议                 | 数据集                   | 备注 |
| ------------------------------------------------------------ | ------ | -------------------- | ------------------------ | ---- |
| Cross-modal Moment Localization in Videos :o:                | ROLE   | MM 2018              | Charades-STA<br />DiDeMo |      |
| Attentive Moment Retrieval in Videos :o:                     | ACRN   | SIGIR 2018           | DiDeMo<br />TACoS        |      |
| Localizing Moments in Video with Temporal Language           |        | EMNLP 2018 ==CCF-B== |                          |      |
| Temporally Grounding Natural Sentence in Video               |        | EMNLP 2018           |                          |      |
| Multi-modal Circulant Fusion for Video-to-Language and Backward |        | IJCAI 2018           |                          |      |



#### Cross-modal Moment Localization in Videos :white_check_mark:

> 《Cross-modal Moment Localization in Videos》
>
> - URL：https://dl.acm.org/doi/10.1145/3240508.3240549
> - Official Code：
>   - Google Driver：https://drive.google.com/drive/folders/1lVxcC4TDGcGtB_tTAp4z6i_WmZHpznc2
>   - https://github.com/Last-Malloc/ROLE
> - 主页：https://acmmm18.wixsite.com/role

强调时间语态问题，即句子中的 “先”、“前” 往往被忽视，需要更细致的理解。

方法：尝试结合 **上下文**，用文本给视频加 **Attention**



<img src="https://static.wixstatic.com/media/496c50_862702e8efcd4626aaf5a77a2aaa4680~mv2.png/v1/fill/w_750,h_383,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/496c50_862702e8efcd4626aaf5a77a2aaa4680~mv2.png" alt="ROLE’s Pipeline" style="zoom:80%;" />





#### Attentive Moment Retrieval in Videos :white_check_mark:

> 《Attentive Moment Retrieval in Videos》
>
> - URL：https://dl.acm.org/doi/abs/10.1145/3209978.3210003
> - 主页：https://sigir2018.wixsite.com/acrn
> - ACRN’s Code：[百度网盘](https://pan.baidu.com/s/1eUgvASi?_at_=1676036385093#list/path=%2F&parentPath=%2F)

方法：尝试结合 **上下文**，用视频给文本加 **Attention**



![ACRN](https://static.wixstatic.com/media/8263ca_1e69cc739e044520b82605b54908e343~mv2_d_2085_1311_s_2.png/v1/fill/w_529,h_329,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/8263ca_1e69cc739e044520b82605b54908e343~mv2_d_2085_1311_s_2.png)

Input: A set of moment candidates and the given query. 

Output: A ranking model mapping each moment-query pair to a relevance score and estimating their location offsets of the golden moment. 

As Figure illustrates, our proposed ACRN model comprises of the following components:

1. The memory attention network leverages the weighting contexts to enhance the visual embedding of each moment;
2. The cross-modal fusion network explores the intra-modal and the inter-modal feature interactions to generate the moment-query representations;

3. The regression network estimates the relevance scores and predicts the location o sets of the golden moments. 



#### Localizing Moments in Video with Temporal Language









### 2019

> [!Info+年度关键词]
>
> **文本和视频的细粒化；强化学习；候选框关系/生成；弱监督**
>
> RL-based、Candidate-based

| 标题                                                         | 方法名                                                       | 会议                | 数据集                              | 备注                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------- | ----------------------------------- | -------------------- |
| Read, Watch, and Move **Reinforcement Learning** for Temporally Grounding Natural Language Descriptions in Videos :o: |                                                              | AAAI 2019           |                                     | 强化学习（首次）     |
| MAN: Moment Alignment Network for Natural Language Moment Retrieval via **Iterative Graph Adjustment** :o: | MAN<br />（Moment Alignment Network）                        | CVPR 2019           | DiDeMo<br />Charades-STA            |                      |
| Language-Driven Temporal Activity Localization A Semantic Matching **Reinforcement Learning** Model :o: |                                                              | CVPR 2019           |                                     | 强化学习             |
| **Weakly Supervised** Video Moment Retrieval From Text Queries :o: | TGA<br />（Text Guided Attention）                           | CVPR 2019           | DiDeMo<br />Charades-STA            | 弱监督<br />（首次） |
| Cross-Modal Video Moment Retrieval with **Spatial and Language-Temporal Attention** :o: | SLTA<br />（**S**patial and **L**anguage-**T**emporal **A**ttention model） | ICMR 2019 ==CCF-B== | TACoS<br />Charades-STA<br />DiDeMo |                      |
|                                                              |                                                              |                     |                                     |                      |
|                                                              |                                                              |                     |                                     |                      |



> 除了 **强化学习** 方法在某种程度上可以直接做到定位，但大多数方法还是 **基于候选集**。
>
> 但是生成的候选集又太多了，所以也有几篇在候选集身上打主意的方法。

#### MAN: Moment Alignment Network for Natural Language Moment Retrieval via Iterative Graph Adjustment :white_check_mark:

> MAN: Moment Alignment Network for Natural Language Moment Retrieval via Iterative Graph Adjustment
>
> - URL：https://arxiv.org/abs/1812.00087
> - Official Code：https://github.com/dazhang-cv/MAN ==TensorFlow==

- **解决问题：**根据自然语言的一句话检索视频所对应的帧片段，并同时考虑到 **语义错位**（e.g. the second time）和 **结构错位**（候选集之间的处理是独立的，e.g. after）。
- **特征抽取**：LSTM 和 I3D
- **创新点**：**Iterative graph adjustment network**，即尝试对所生成的各个候选片段构图，并设计一个迭代图调整网络来 **学习候选时刻之间的关系**。

MAN 模型如下所示，主要包括三部分：

- a language encoder
- a video encoder
- IGAN (an iterative graph adjustment network)

![MAN](https://github.com/dazhang-cv/MAN/blob/master/man.png?raw=true)





#### Weakly Supervised Video Moment Retrieval From Text Queries :white_check_mark:

> Weakly Supervised Video Moment Retrieval From Text Queries
>
> - URL：https://arxiv.org/abs/1904.03282
> - Official Code：https://github.com/niluthpol/weak_supervised_video_moment ==PyTorch==
> - DeepAI：https://deepai.org/publication/weakly-supervised-video-moment-retrieval-from-text-queries

- 动机：标注句子边界太耗时，且在实践中不可扩展

- 方法：通过 **弱监督** 的方式，利用 **Text Guided Attention（TGA）** 来学习 **隐式对齐** 文本-视频对 中的文本与视频，而不是直接学习文本描述对应的时间边界（起始、终止时间）。
  - 只使用视频级别的文本描述
  - 减少了手动标注数据带来的问题，能够更容易的获取标注的数据。

![弱监督方法的整体框架](https://images.deepai.org/converted-papers/1904.03282/x2.png)



![Text Guided Attention](https://images.deepai.org/converted-papers/1904.03282/x3.png)



#### Cross-Modal Video Moment Retrieval with Spatial and Language-Temporal Attention :white_check_mark:

> Cross-Modal Video Moment Retrieval with Spatial and Language-Temporal Attention
>
> - URL：https://dl.acm.org/doi/10.1145/3323873.3325019
> - Official Code：https://github.com/BonnieHuangxin/SLTA ==TensorFlow==
> - 主页：https://icmr2019.wixsite.com/slta
> - 知乎博客：https://zhuanlan.zhihu.com/p/64852440

> 识别关键字、最相关物体，并关注物体间的交互

- 动机：没有关注空间目标信息和文本的语义对齐，且需要结合视频的时空信息。
- 方法：
  - 特征抽取：全局 + 局部，即全局用 C3D 和 BiLSTM；然后全局特征分别给局部（词和目标）加 attention 突出重点。

**S**patial and **L**anguage-**T**emporal **A**ttention model（**SLTA**）即 **空间与语言-时序注意力**。它包括 **两个分支注意力网络**，分别为 **空间注意力、语言-时序注意力**。

- 首先，我们提取视频帧 object-level 的局部特征，并通过 **空间注意力** 来关注与 query 最相关的局部特征（例如，局部特征 “girl”，“cup”），然后对连续帧上的局部特征序列进行 encoding，以捕获这些 object 之间的交互信息（例如，涉及这两个 object 的交互动作 “pour”）

- 同时，利用 **语言-时序注意力网络** 基于视频片段上下文信息来强调 query 中的关键词

因此，我们提出的 **两个注意力子网络** 可以 **识别视频中最相关的物体** 和 **物体间的交互**，同时 **关注 query 中的关键字**。

![SLTA](https://static.wixstatic.com/media/6e3fbf_07678b8b46e1473f85f3b0aab4ac4581~mv2_d_3543_1380_s_2.jpg/v1/fill/w_1221,h_473,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/Figure2.jpg)





#### ABLR

> 《To Find Where You Talk: Temporal Sentence Localization in Video with Attention Based Location Regression》
>
> - URL：https://arxiv.org/abs/1804.07014
> - Code：https://github.com/yytzsy/ABLR_code

这一篇论文主要有两个亮点：

- 去掉了滑动窗口算法；
- 引入 attention 进行跨模态信息的交互；

![](https://pic1.zhimg.com/80/v2-f20e37d45621452ad37de88ed29f6710_720w.webp)



#### ACL

> 《MAC: Mining Activity Concepts for Language-based Temporal Localization》
>
> - URL：https://arxiv.org/abs/1811.08925
> - Code：https://github.com/runzhouge/MAC
> - 会议：

这篇文章的方法是建立在 CTRL 模型之上的。它的亮点在于强化了两个模态之间动作语义信息的对应，提出了 Activity Concepts based Localizer（ACL）模型，模型结构如下图所示：

![](https://pic2.zhimg.com/80/v2-3354a191d85e3e6b514dfb74b44ac475_720w.webp)

模型相比 CTRL 有了如下改变：

- video 不是直接滑窗生成候选片段，而是先以若干帧为单元得到单元级别的特征，再以连续单元得到候选片段的特征
  - 视频单元特征的抽取针对不同数据集有使用 C3D 和I3D

- 引入 Activity Concept。作者提前准备一个列表，是视频描述中所有涉及的动名词组。候选片段特征和句子特征分别还有对应的动作语义特征（对应上图 Visual Activity Concept 和 Semantic Activity Concept）

- 句子描述和视频除了特征上的匹配，还加上了两者动作语义上的特征匹配。两个匹配信息融合，作为最后对齐 loss 和定位回归 loss 的输入

- 由于并不是所有视频段都存在动作语义概念，每个候选视频段还有一个 **Actionness score 用来衡量该候选视频段是否存在动作**
  - 这个 score 有用（在求 loss 的时候）





### 2020（3/22）

| 标题                                                         | 方法名 | 会议      | 数据集                                            | 备注     |
| ------------------------------------------------------------ | ------ | --------- | ------------------------------------------------- | -------- |
| Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language :o: | 2D-TAN | AAAI 2020 | Charades-STA<br />TACoS<br />ActivityNet Captions |          |
| Moment Retrieval via Cross-Modal Interaction Networks With **Query Reconstruction** :o: |        | TIP 2020  |                                                   |          |
| **Adversarial** Video Moment Retrieval by Jointly Modeling Ranking and Localization :o: | AVMR   | MM 2020   | Charades-STA<br />TACoS                           | 对抗训练 |
|                                                              |        |           |                                                   |          |
|                                                              |        |           |                                                   |          |
|                                                              |        |           |                                                   |          |



#### Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language

> Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language
>
> - URL：https://arxiv.org/abs/1912.03590
> - Official Code：https://github.com/microsoft/VideoX/tree/master/2D-TAN
> - Others Code：[optimized implementation](https://github.com/ChenJoya/2dtan)

> 参考博客：
>
> - 知乎：[ICCV'2019 视频内容片段定位冠军方案 2D-TAN 的一个优化实现（内附论文解读）](https://zhuanlan.zhihu.com/p/147779357)
> - CSDN：[【论文阅读】Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language](https://blog.csdn.net/YasmineC/article/details/123176521)

出自 AAAI 2020，Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language，2019 ICCV HACS 动作定位挑战赛冠军方案。

- Motivation：以往模型一般单独考虑时间，忽视了时间的依赖性。（不能在当前时刻对其他时刻进行预测）
- Method：将 **定位起始解耦**，**时间从一维变为二维**，即变成二维矩阵，预测每一个位置的 score。这样做的好处是特征图中每个位置的视频特征都与文本特征融合，得到相似度特征图（图像中左侧的绿色立方体），而这个矩阵的大小可以直接通过切分视频的细粒度做到，然后将融合后的相似性特征映射通过一系列卷积层，并逐层建立各段与其周围段之间的关系，最后将考虑邻域关系的相似度特征输入到完全连通层中，得到最终的得分。

##### 1）二维时间图

目前，定位视频中的某一片段大多采用以不同大小，交叠比的 sliding window 对于一个视频产生诸多的 proposals（候选者），再对这些 proposals 进行分类与回归。

2D-TAN 中提出的二维时间图可以看成一种 “优雅的 sliding window”：如下图所示，以两个轴分别代表开始时间段和结束时间段，构建 $N\times N$ 的 2D map。

> 这里的一段 $\tau = duration / N$，$duration$ 是整个视频的时长，$N$ 为 2D map 的大小。

<img src="https://pic2.zhimg.com/80/v2-f31c7351703599903aa5dcf9b9f89e85_720w.webp" alt="二维时间图" style="zoom: 67%;" />

这张 2D map 里的 **每一个 cell 都代表着一个时间片段**，且只有 **右上角半区** 的时间片段为有效的（起始时间<结束时间）。

那么，我们只需要判断每个 cell 对应的时间与给定 query 的匹配度即可。为了降低计算量，作者设计了 **采样机制**，只对上方蓝色点的 cell 进行训练/推理。



#### 2）视觉-语言特征融合

- 2D map 可以用来承载视频的视觉特征，每一个 cell 对应的 $d^V$ 维视频特征填入到 2D map 中，然后我们就可以得到 $N\times N\times d^V$ 的 2D feature map。

- 接着，将查询 query 中的每个文本单词转换成 Glove 词向量，再将词向量依次通过 LSTM 网络，使用其最后一层的输出作为文本语句的特征。

- 然后，将文本语句特征与 2D feature map 的每一个 cell 进行点乘（Hadamard Product），即可得到视觉-语言融合后的 2D feature map。

![2D-TAN](https://pic3.zhimg.com/80/v2-d3631339fdee9332f530021f9047a156_720w.webp)



#### 3）时间邻近卷积

> 这部分在上述结构图上未曾详细表现出来（只有一个简单的 **ConvNet** 字样），但是个人认为它才是文章的核心思想所在。

前人的工作都是 **独立地匹配句子和一个片段**，而忽略了其他片段对其影响。例如，当我们要定位 “这个人又吹起了萨克斯 The guy plays the saxophone again”，如果只看后面的视频而不看前面的，我们不可能在视频中定位到这个片段。

我们知道，卷积网络是具有感受野的，而 2D feature map 的形式由非常适用于卷积，因此作者在这里加入 **多层卷积** 再次得到 2D feature map，而 **此时的每一个 cell 都将具有邻近 cell 的信息，而非单独考虑每个片段**，可以学习更具有区分性的特征。

<img src="https://pic3.zhimg.com/80/v2-2bc9a209f454f4f10a3fd1b0c899d5b2_720w.webp" alt="时间邻近卷积" style="zoom:67%;" />



#### 4）小问题

- 二维图上的点扩展到向量呢？
  - 直觉来看，由点变成向量，反而没那么灵活了
- 穷举地更详尽了，为什么反而计算成本降低了呢？
  - 因为每个片段都被降维了
  - 这样能把关键信息展示出来吗？

- 是不是说最后的 **检索误差就是 $\tau$** 呢？那么误差还挺大的。
- $\tau$ 是怎么得到的呢？每个视频所对应的 $\tau$ 是否一样？



#### 5）更多：MS-2D-TAN（新的改进）

作者对于 2D-TAN 的改进：MS-2D-TAN

![MS-2D-TAN](https://github.com/microsoft/VideoX/raw/master/MS-2D-TAN/pipeline.jpg)



#### Moment Retrieval via Cross-Modal Interaction Networks With Query Reconstruction

> Moment Retrieval via Cross-Modal Interaction Networks With Query Reconstruction
>
> - URL：https://ieeexplore.ieee.org/abstract/document/8962274
> - Code：



![Moment retrieval in video](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/83/8835130/8962274/zhao1-2965987-large.gif)



![The framework of cross-modal interaction networks for moment retrieval](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/83/8835130/8962274/zhao2-2965987-large.gif)



![The details of cross-modal interaction networks for moment retrieval](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/83/8835130/8962274/zhao3abcde-2965987-large.gif)



#### Adversarial Video Moment Retrieval by Jointly Modeling Ranking and Localization

> Adversarial Video Moment Retrieval by Jointly Modeling Ranking and Localization
>
> - URL：https://dl.acm.org/doi/abs/10.1145/3394171.3413841
> - Official Code：https://github.com/yawenzeng/AVMR ==PyTorch==

- 动机：强化学习定位不稳定，检索任务候选框效率低。
- 方法：用对抗学习联合建模，即用强化学习定位来生成候选，进而在生成候选集上进行检索。

![AVMR](https://img-blog.csdnimg.cn/20200729154432695.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5Mzg4NDEw,size_16,color_FFFFFF,t_70#pic_center)



该作者的扩展论文也整理了一些相关 RL-based 定位方法和 Candidate-based 检索方法的表格：

![一些 RL-based 与 Candidate-based 的总结](https://img-blog.csdnimg.cn/c86ff0ece1084c6a959c26863da4a8ef.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5Mzg4NDEw,size_16,color_FFFFFF,t_70)

> 图片来源：[TOMM2021] Moment is Important: Language-Based Video Moment Retrieval via Adversarial Learning



### 2021（2/11）

| 标题                                                         | 方法名 | 会议      | 数据集 | 备注 |
| ------------------------------------------------------------ | ------ | --------- | ------ | ---- |
| Multi-Modal Relational Graph for Cross-Modal Video Moment Retrieval :o: |        | CVPR 2021 |        |      |
| A Closer Look at Temporal Sentence Grounding in Videos_ Datasets and Metrics :o: |        |           |        |      |
|                                                              |        |           |        |      |
|                                                              |        |           |        |      |



#### Multi-Modal Relational Graph for Cross-Modal Video Moment Retrieval

> Multi-Modal Relational Graph for Cross-Modal Video Moment Retrieval
>
> - 





#### A Closer Look at Temporal Sentence Grounding in Videos_ Datasets and Metrics

> A Closer Look at Temporal Sentence Grounding in Videos_ Datasets and Metrics
>
> - 





### 2022

| 标题                                                         | 方法名 | 会议       | 数据集 | 备注                          |
| ------------------------------------------------------------ | ------ | ---------- | ------ | ----------------------------- |
| A Survey on Temporal Sentence Grounding in Videos :o:        |        |            |        | 综述                          |
| The Elements of Temporal Sentence Grounding in Videos: A Survey and Future Directions :o: |        |            |        | 综述                          |
| Unsupervised Temporal Video Grounding with Deep Semantic Clustering :o: |        | AAAI 2022  |        | 无监督学习                    |
| Point Prompt Tuning for Temporally Language Grounding :o:    |        | SIGIR 2022 |        | Prompt-Learning<br />（首次） |
|                                                              |        |            |        |                               |
|                                                              |        |            |        |                               |
|                                                              |        |            |        |                               |



#### A Survey on Temporal Sentence Grounding in Videos





#### The Elements of Temporal Sentence Grounding in Videos: A Survey and Future Directions







#### Unsupervised Temporal Video Grounding with Deep Semantic Clustering

> Unsupervised Temporal Video Grounding with Deep Semantic Clustering
>
> - URL：
> - Code：





#### Point Prompt Tuning for Temporally Language Grounding

> Point Prompt Tuning for Temporally Language Grounding
>
> - URL：
> - Code：








- [x] Partially Relevant Video Retrieval <mark>CCF-A</mark>
    - ACM MM 2022 oral
    - [github](https://github.com/HuiGuanLab/ms-sl)
    - [paper's url](https://arxiv.org/abs/2208.12510)
    - 参考：<a href="https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/127505397">CSDN：ACM MM 2022 Oral | PRVR：全新的文本到视频跨模态检索子任务</a>

- [ ] a







## Video-Video Retrival

基于视频/图片查询的视频片段检索任务

### 2019

| 标题 | 方法名 | 会议 | 数据集 | 备注 |
| ---- | ------ | ---- | ------ | ---- |
|      |        |      |        |      |
|      |        |      |        |      |




### 2020

| 标题 | 方法名 | 会议 | 数据集 | 备注 |
| ---- | ------ | ---- | ------ | ---- |
|      |        |      |        |      |
|      |        |      |        |      |


### 2021

| 标题 | 方法名 | 会议 | 数据集 | 备注 |
| ---- | ------ | ---- | ------ | ---- |
|      |        |      |        |      |
|      |        |      |        |      |



#### Clip4clip







### 2022

| 标题 | 方法名 | 会议 | 数据集 | 备注 |
| ---- | ------ | ---- | ------ | ---- |
|      |        |      |        |      |
|      |        |      |        |      |





## 参考



- HNU 的 yawenzeng 程序媛（Tencent Inc.）
  - github：[Awesome-Cross-Modal-Video-Moment-Retrieval](https://github.com/nakaizura/Awesome-Cross-Modal-Video-Moment-Retrieval)
  - CSDN：[Cross-modal Video Moment Retrieval（跨模态视频时刻检索综述）](https://blog.csdn.net/qq_39388410/article/details/107316185)
  
- 部分 SOT 方法的代码：[Temporally-language-grounding](https://github.com/WuJie1010/Temporally-language-grounding)

- Awesome：[Temporal Language Grounding](https://github.com/SCZwangxiao/Temporal-Language-Grounding-in-videos)
  - 知乎：[【综述】Temporal Sentence Grounding](https://zhuanlan.zhihu.com/p/101555506)









text-video retrival：

video-video retrival：有两种想法

- 直接法：query video -> video
- 间接法：query video -> query caption/text -> video
  - query video 相比于 video，长度短很多（不是一个数量级），可以使用更轻量化/更窄的模型来提取特征



---

特征提取：
文本：采用基于预训练好的 Transformer 的模型（BERT、RoBERTa）
视频：看数据集，一般使用 3D Conv 进行提取


多尺度的双流 Transformer 模态交互
不仅句子可以调节、选择视频；视频也可以反向选择和调节查询句子；
通常情况下，视频包含时序性、表现型、光流性、声音等比自然语言更加丰富的信息，所以仅靠一次交互很难捕捉到想要的内容；
多尺度视频级别 Features（类似 MS-2D-TAN）；片段级别 Features；
Video-Query Attention
Query-VIdeo Attention

结果预测：:o:
**同时结合 Prompt-Learning 和 Query-Reconstruction（Recaption）来进行视频时刻定位**
**通过模型预测的**
**既能够同时使用有监督、弱监督的数据集，同时又能够保留（maintain）现有的有监督数据集提供的准确性（尽管数据集中存在偏差/误差）**




如何使用目前预训练好的多模态大模型的能力？

Proposal-based：
Sliding Windows（SW）
Proposa Generated（PG）





