---
layout: post
author: bookstall
tags: Transformer
categories: [Transformer]
excerpt: 介绍各种 Transformer 位置编码方式，包括：
keywords: Transformer
title: Transformer 的各种位置编码方式
mathjax: true
---

## 位置编码

不同于 RNN、CNN 等模型，对于 Transformer 模型来说，位置编码的加入是 **必不可少** 的，因为 **纯粹的 Attention 模块是无法捕捉输入顺序的**，即 **无法区分不同位置的 Token**。为此我们大体有两个选择：

- 想办法 <u>将位置信息融入到输入中</u>，这构成了 **绝对位置编码** 的一般做法；

- 想办法 <u>微调一下 Attention 结构</u>，使得它有能力分辨不同位置的 Token，这构成了 **相对位置编码** 的一般做法。

虽然说起来主要就是绝对位置编码和相对位置编码两大类，但每一类其实又能衍生出各种各样的变种，为此研究人员可算是煞费苦心、绞尽脑汁了，此外还有一些不按套路出牌的位置编码。

本文就让我们来欣赏一下研究人员为了更好地表达位置信息所构建出来的 “八仙过海，各显神通” 般的编码方案。

### 1、绝对位置编码

形式上来看，绝对位置编码是相对简单的一种方案，但即便如此，也不妨碍各路研究人员的奇思妙想，也有不少的变种。一般来说，绝对位置编码会加到输入中：在输入的第 $$k$$ 个向量 $$x_k$$ 中加入位置向量 $$p_k$$ 变为 $$x_k+p_k$$，其中 $$p_k$$ 只依赖于位置编号 $$k$$。


#### 1.1、训练式（可学习的）

很显然，绝对位置编码的一个最朴素方案是不特意去设计什么，而是 **直接将位置编码当作可训练参数**，比如最大长度为 512，编码维度为 768，那么就初始化一个 $$512\times 768$$ 的矩阵作为位置向量，让它随着训练过程更新。现在的 BERT、GPT 等模型所用的就是这种位置编码，事实上它还可以追溯得更早，比如 2017 年 Facebook 的 [《Convolutional Sequence to Sequence Learning》](https://arxiv.org/abs/1705.03122) 就已经用到了它。

对于这种训练式的绝对位置编码，一般的认为它的缺点是 **没有外推性**，即如果预训练最大长度为 512 的话，那么最多就只能处理长度为 512 的句子，再长就处理不了了。当然，也可以将超过 512 的位置向量随机初始化，然后继续微调。

但笔者最近的研究表明，通过 **层次分解** 的方式，可以使得绝对位置编码能外推到足够长的范围，同时保持还不错的效果，细节请参考笔者之前的博文 [《层次分解位置编码，让BERT可以处理超长文本》](https://kexue.fm/archives/7947)。因此，**其实外推性也不是绝对位置编码的明显缺点**。

![位置编码的层次分解示意图](https://kexue.fm/usr/uploads/2020/12/2058846820.png)


#### 1.2、三角式：Sinusoidal 位置编码

三角函数式位置编码，一般也称为 **Sinusoidal 位置编码**，是 Google 的论文 [《Attention is All You Need》](https://arxiv.org/abs/1706.03762) 所提出来的一个显式解：

$$
\begin{equation}
\left\{\begin{aligned}&\boldsymbol{p}_{k,2i}=\sin\Big(k/10000^{2i/d}\Big)\\ 
&\boldsymbol{p}_{k, 2i+1}=\cos\Big(k/10000^{2i/d}\Big) 
\end{aligned}\right.
\end{equation}
$$

其中 $$p_{k,2i}, p_{k,2i+1}$$ 分别是位置 $$k$$ 的编码向量的第 $$2i,2i+1$$ 个分量，$$d$$ 是位置向量的维度。

很明显，三角函数式位置编码的特点是有显式的生成规律，因此可以期望于它 **有一定的外推性**。

另外一个使用它的理由是：由于

$$
\sin(α+β)=\sinα\cosβ+\cosα\sinβ
$$ 

以及 

$$
\cos(α+β)=\cosα\cosβ−\sinα\sinβ
$$

这表明位置 $$α+β$$ 的向量可以表示成位置 $$α$$ 和位置 $$β$$ 的向量组合，这 **提供了表达相对位置信息的可能性**。

但很奇怪的是，现在我们很少能看到直接使用这种形式的绝对位置编码的工作，原因不详。


#### 1.3、递归式：RNN 模型

原则上来说，**RNN 模型** 不需要位置编码，它在结构上就**自带了学习到位置信息的可能性**（因为递归就意味着我们可以训练一个 **“数数” 模型**），因此，<u>如果在输入后面先接一层 RNN，然后再接 Transformer，那么理论上就不需要加位置编码了</u>。同理，我们也可以用 RNN 模型来学习一种绝对位置编码，比如从一个向量 $$p_0$$ 出发，通过递归格式 $$p_{k+1}=f(p_k)$$ 来得到各个位置的编码向量。

ICML 2020 的论文 [《Learning to Encode Position for Transformer with Continuous Dynamical Model》](https://arxiv.org/abs/2003.09229) 把这个思想推到了极致，它提出了用微分方程（ODE）$$d\boldsymbol{p}_t/dt=\boldsymbol{h}(\boldsymbol{p}_t,t)$$ 的方式来建模位置编码，该方案称之为 **FLOATER**。显然，FLOATER 也属于递归模型，函数 $$h(p_t,t)$$ 可以通过神经网络来建模，因此这种微分方程也称为 **神经微分方程**，关于它的工作最近也逐渐多了起来。

理论上来说，基于递归模型的位置编码也具有比较好的外推性，同时它也比三角函数式的位置编码有 **更好的灵活性**（比如容易证明 **三角函数式的位置编码就是 FLOATER 的某个特解**）。但是很明显，递归形式的位置编码牺牲了一定的并行性，**可能会带速度瓶颈**。


#### 1.4、相乘式：乘性位置编码

刚才我们说到，输入 $$x_k$$ 与绝对位置编码 $$p_k$$ 的组合方式一般是 $$x_k+p_k$$，那有没有 “不一般” 的组合方式呢？比如 $$x_k⊗p_k$$（逐位相乘）？我们平时在搭建模型的时候，对于融合两个向量有多种方式，相加、相乘甚至拼接都是可以考虑的，怎么大家在做绝对位置编码的时候，都默认只考虑相加了？

很抱歉，笔者也不知道答案。可能大家默认选择相加是因为向量的相加具有比较鲜明的几何意义，但是对于深度学习模型来说，这种几何意义其实没有什么实际的价值。

最近笔者看到的一个实验显示，似乎将 “加” 换成 “乘”，也就是 $$x_k⊗p_k$$ 的方式，似乎比 $$x_k+p_k$$ 能取得更好的结果。具体效果笔者也没有完整对比过，只是提供这么一种可能性。关于实验来源，可以参考 [《中文语言模型研究：(1) 乘性位置编码》](https://zhuanlan.zhihu.com/p/183234823)。

> 这里的乘性位置编码，本质上就是对 **Softmax 之后的 Attention** 进行 **加权** 操作！ 


### 2、相对位置编码

相对位置并没有完整建模每个输入的位置信息，而是 **在算 Attention 的时候考虑当前位置与被 Attention 的位置的相对距离**，由于自然语言一般更依赖于相对位置，所以相对位置编码通常也有着优秀的表现。对于相对位置编码来说，它的灵活性更大，更加体现出了研究人员的 “天马行空”。

#### 2.1、经典式

相对位置编码起源于 Google 的论文 [《Self-Attention with Relative Position Representations》](https://arxiv.org/abs/1803.02155)，华为开源的 NEZHA 模型也用到了这种位置编码，后面各种相对位置编码变体基本也是依葫芦画瓢的简单修改。

一般认为，**相对位置编码是由绝对位置编码启发而来**，考虑一般的 **带绝对位置编码的 Attention**：

$$
\begin{equation}
\left\{\begin{aligned} 
\boldsymbol{q}_i =&\, (\boldsymbol{x}_i + \boldsymbol{p}_i)\boldsymbol{W}_Q \\ 
\boldsymbol{k}_j =&\, (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_K \\ 
\boldsymbol{v}_j =&\, (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_V \\ 
a_{i,j} =&\, softmax\left(\boldsymbol{q}_i \boldsymbol{k}_j^{\top}\right)\\ 
\boldsymbol{o}_i =&\, \sum_j a_{i,j}\boldsymbol{v}_j 
\end{aligned}\right.
\end{equation}
$$

其中softmax
对j
那一维归一化，这里的向量都是指行向量。我们初步展开qik⊤j
：

$$
\begin{equation}
\color{red}{
\boldsymbol{q}_i \boldsymbol{k}_j^{\top} = \left(\boldsymbol{x}_i + \boldsymbol{p}_i\right)\boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\left(\boldsymbol{x}_j + \boldsymbol{p}_j\right)^{\top} = \left(\boldsymbol{x}_i \boldsymbol{W}_Q + \boldsymbol{p}_i \boldsymbol{W}_Q\right)\left(\boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{W}_K^{\top}\boldsymbol{p}_j^{\top}\right) 
}
\end{equation}
$$

为了引入相对位置信息，Google 把第一项位置 $$\boldsymbol{p}_i \boldsymbol{W}_Q$$ 去掉，并且第二项 $$p_j W_K$$ 改为二元位置向量 $$R^K_{i,j}$$，变成：

$$
\begin{equation} 
a_{i,j} = softmax\left(\boldsymbol{x}_i \boldsymbol{W}_Q\left(\boldsymbol{x}_j\boldsymbol{W}_K + \color{green}{\boldsymbol{R}_{i,j}^K}\right)^{\top}\right) 
\end{equation}
$$

并且将 $$\color{red}{\boldsymbol{o}_i =\sum\limits_j a_{i,j}\boldsymbol{v}_j = \sum\limits_j a_{i,j}(\boldsymbol{x}_j\boldsymbol{W}_V + \boldsymbol{p}_j\boldsymbol{W}_V)}$$ 中的 $$\boldsymbol{p}_j \boldsymbol{W}_V$$ 换成 $$\boldsymbol{R}_{i,j}^{V}$$，即：

$$
\begin{equation}
\boldsymbol{o}_i = \sum_j a_{i,j}\left(\boldsymbol{x}_j\boldsymbol{W}_V + \color{green}{\boldsymbol{R}_{i,j}^{V}}\right) 
\end{equation}
$$

---

所谓相对位置，是将本来依赖于二元坐标 $$(i,j)$$ 的向量 $$R^K_{i,j},R^V_{i,j}$$，改为只依赖于相对距离 $$i−j$$，并且通常来说会进行截断，以适应不同任意的距离，即：

$$
\begin{equation}\label{eq:rp-clip}
\begin{aligned} 
\boldsymbol{R}_{i,j}^{K} = \boldsymbol{p}_K\left[\text{clip}(i-j, p_{\min}, p_{\max})\right]\\ 
\boldsymbol{R}_{i,j}^{V} = \boldsymbol{p}_V\left[\text{clip}(i-j, p_{\min}, p_{\max})\right] 
\end{aligned}
\end{equation}
$$

这样一来，**只需要有限个位置编码，就可以表达出任意长度的相对位置（因为进行了截断）**，不管 $$p_K,p_V$$ 是选择可训练式的还是三角函数式的，都可以达到处理任意长度文本的需求。


#### 2.2、XLNET 式

XLNET 式位置编码其实源自 Transformer-XL 的论文 [《Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context》](https://arxiv.org/abs/1901.02860)，只不过因为使用了 Transformer-XL 架构的 XLNET 模型并在一定程度上超过了 BERT 后，Transformer-XL 才算广为人知，因此这种位置编码通常也被冠以 XLNET 之名。

XLNET 式位置编码源于对上述 $$q_i k^⊤_j$$ 的完全展开：

$$
\begin{equation}\label{eq:qk-exp} 
\boldsymbol{q}_i \boldsymbol{k}_j^{\top} = \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{p}_j^{\top} + \boldsymbol{p}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{p}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{p}_j^{\top}
\end{equation}
$$

Transformer-XL 的做法很简单，直接将 $$p_j$$ 替换为相对位置向量 $$R_{i−j}$$，至于两个 $$p_i$$，则干脆替换为两个可训练的向量 $$u,v$$：

$$
\begin{equation}
\boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\color{green}{\boldsymbol{R}_{i-j}^{\top}} +  \color{red}{\boldsymbol{u}}\boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \color{red}{\boldsymbol{v}} \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\color{green}{\boldsymbol{R}_{i-j}^{\top}} 
\end{equation}
$$

该编码方式中的 $$R_{i−j}$$ 没有像式 $$\eqref{eq:rp-clip}$$ 那样进行截断，而是直接用了 Sinusoidal 式的生成方案。

此外，$$v_j$$ 上的位置偏置就直接去掉了，即直接令 $$\boldsymbol{o}_i = \sum\limits_j a_{i,j}\boldsymbol{x}_j\boldsymbol{W}_V$$。**似乎从这个工作开始，后面的相对位置编码都只加到 Attention 矩阵上去，而不加到 $$v_j$$ 上去了。**


#### 2.3、T5 式

T5 模型出自文章 [《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》](https://arxiv.org/abs/1910.10683)，里边用到了一种更简单的相对位置编码。思路依然源自展开式 $$\eqref{eq:qk-exp}$$，如果非要分析每一项的含义，那么可以分别理解为 **“输入-输入”、“输入-位置”、“位置-输入”、“位置-位置”** 四项注意力的组合。

如果我们认为 **输入信息与位置信息应该是独立（解耦）的**，那么它们就不应该有过多的交互，所以 **“输入-位置”、“位置-输入”** 两项 Attention 可以删掉，而 $$p_i W_Q W^⊤_K p^⊤_j$$ 实际上只是一个只依赖于 $$(i,j)$$ 的标量，我们可以直接将它作为参数训练出来，即简化为：

$$
\begin{equation}
\boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \color{green}{\boldsymbol{\beta}_{i,j}}
\end{equation}
$$

说白了，它仅仅是在 Attention 矩阵的基础上 **加一个可训练的偏置项** 而已，而跟 XLNET 式一样，在 $$v_j$$ 上的位置偏置则直接被去掉了。包含同样的思想的还有微软在 ICLR 2021 的论文 [《Rethinking Positional Encoding in Language Pre-training》](https://arxiv.org/abs/2006.15595) 中提出的 TUPE 位置编码。

比较 “别致” 的是，不同于常规位置编码对将 $$β_{i,j}$$ 视为 $$i−j$$ 的函数并进行截断的做法，T5 对相对位置进行了一个 **“分桶”** 处理，即相对位置是 $$i−j$$ 的位置实际上对应的是 $$f(i−j)$$ 位置，映射关系如下：

$$
\begin{array}{c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c} 
\hline 
i - j & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 & 14 & 15\\ 
\hline 
f(i-j) & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 8 & 8 & 8 & 9 & 9 & 9 & 9 \\ 
\hline 
i - j & 16 & 17 & 18 & 19 & 20 & 21 & 22 & 23 & 24 & 25 & 26 & 27 & 28 & 29 & 30 & \cdots\\ 
\hline 
f(i-j) & 10 & 10 & 10 & 10 & 10 & 10 & 10 & 11 & 11 & 11 & 11 & 11 & 11 & 11 & 11 & \cdots \\ 
\hline\end{array}
$$

具体的映射代码，读者自行看源码就好。这个设计的思路其实也很直观，就是比较邻近的位置（0～7），我们需要比较得精细一些，所以给它们都分配一个独立的位置编码，至于稍远的位置（比如8～11），我们不用区分得太清楚，所以它们可以共用一个位置编码，**距离越远，共用的范围就可以越大，直到达到指定范围再 clip**。


#### 2.4、DeBERTa 式

DeBERTa 也是微软搞的，2020 年 6 月就发出来了，论文为 [《DeBERTa: Decoding-enhanced BERT with Disentangled Attention》](https://arxiv.org/abs/2006.03654)，最近又小小地火了一把，一是因为它正式中了 ICLR 2021，二则是它登上 [SuperGLUE](https://super.gluebenchmark.com/) 的榜首，成绩稍微超过了 T5。

其实 DeBERTa 的主要改进也是在位置编码上，同样还是从展开式 $$\eqref{eq:qk-exp}$$ 出发，T5 是干脆去掉了第 2、3 项，只保留第 4 项并替换为相对位置编码，而 DeBERTa 则刚刚相反，它扔掉了第 4 项，保留第 2、3 项并且替换为相对位置编码（果然，<u>科研就是枚举所有的排列组合看哪个最优</u>）：

$$
\begin{equation} 
\boldsymbol{q}_i \boldsymbol{k}_j^{\top} = \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\color{green}{\boldsymbol{R}_{i,j}^{\top}} + \color{green}{\boldsymbol{R}_{j,i}} \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} 
\end{equation}
$$

至于 $$R_{i,j}$$ 的设计也是像式 $$\eqref{eq:rp-clip}$$ 那样进行截断的，没有特别的地方。

不过，DeBERTa 比较有意思的地方，是提供了 **使用相对位置和绝对位置编码的一个新视角**，它指出 NLP 的大多数任务可能都只需要相对位置信息，但确实有些场景下绝对位置信息更有帮助，于是它将整个模型分为两部分来理解。以 Base 版的 MLM 预训练模型为例，它一共有 13 层，<u>前 11 层只是用相对位置编码</u>，这部分称为 Encoder，<u>后面 2 层加入绝对位置信息</u>，这部分它称之为 Decoder，还弄了个简称 EMD（Enhanced Mask Decoder）；至于下游任务的微调截断，则是使用前 11 层的 Encoder 加上 1 层的 Decoder 来进行。

> SuperGLUE 上的成绩肯定了 DeBERTa 的价值，但是它论文的各种命名真的是让人觉得极度不适，比如它自称的 “Encoder”、“Decoder” 就很容易让人误解这是一个 Seq2Seq 模型，比如 EMD 这个简称也跟 Earth Mover's Distance 重名。虽然有时候重名是不可避免的，但它重的名都是 ML 界大家都比较熟悉的对象，相当容易引起误解，真不知道作者是怎么想的...



### 3、其他位置编码

绝对位置编码和相对位置编码虽然花样百出，但仍然算是经典范围内，从上述介绍中我们依然可以体会到满满的套路感。除此之外，还有一些并不按照常规套路出牌，它们同样也表达了位置编码。

#### 3.1、CNN 式

尽管经典的将 CNN 用于 NLP 的工作 [《Convolutional Sequence to Sequence Learning》](https://arxiv.org/abs/1705.03122) 往里边加入了位置编码，但我们知道 **一般的 CNN 模型尤其是图像中的 CNN 模型，都是没有另外加位置编码的**，那 CNN 模型究竟是怎么捕捉位置信息的呢？

如果让笔者来回答，那么答案可能是卷积核的各向异性导致了它能分辨出不同方向的相对位置。不过 ICLR 2020 的论文 [《How Much Position Information Do Convolutional Neural Networks Encode?》](https://arxiv.org/abs/2001.08248) 给出了一个可能让人比较意外的答案：**CNN 模型的位置信息，是 Zero Padding 泄漏的**！

我们知道，为了使得卷积编码过程中的 feature 保持一定的大小，我们通常会对输入 padding 一定的 0，而这篇论文显示该操作导致模型有能力识别位置信息。也就是说，卷积核的各向异性固然重要，但是最根本的是 zero padding 的存在，那么可以想象，实际上提取的是当前位置与 padding 的边界的相对距离。

不过，这个能力依赖于 CNN 的局部性，**像 Attention 这种全局的无先验结构并不适用**，如果只关心 Transformer 位置编码方案的读者，这就权当是扩展一下视野吧。

> - **各向异性** 是指一个系统在不同方向上具有不同的性质或响应。例如，在图像处理中，如果一个对象只能在特定的方向上被正确识别，那么它就具有各向异性。举个例子，考虑一个鸟巢的图像。如果我们将图像旋转一定角度，卷积神经网络可能无法正确地识别鸟巢，因为它对于不同的方向有不同的响应。
> 
> - 相反，**各向同性** 指的是一个系统在所有方向上都具有相同的性质或响应。在图像处理中，如果一个对象可以在任何方向上被正确识别，那么它就具有各向同性。举个例子，考虑一个圆形的图像。如果我们将图像旋转任何角度，卷积神经网络仍然能够正确地识别圆形，因为它对于所有方向都有相同的响应。

#### 3.2、复数式

复数式位置编码可谓是最特立独行的一种位置编码方案了，它来自ICLR 2020的论文《Encoding word order in complex embeddings》。论文的主要思想是结合复数的性质以及一些基本原理，推导出了它的位置编码形式（Complex Order）为：


$$
\begin{equation}
\left[r_{j, 1} e^{\text{i}\left(\omega_{j, 1} k+\theta_{j, 1}\right)}, \ldots, r_{j, 2} e^{\text{i}\left(\omega_{j, 2} k+\theta_{j, 2}\right)}, \cdots, r_{j, d} e^{\text{i}\left(\omega_{j, d} k+\theta_{j, d}\right)}\right]\label{eq:complex}
\end{equation}
$$

这里的 $$i$$ 是虚数单位，$$j$$ 代表某个词，$$k$$ 代表该词所在的位置，而

$$
\begin{equation}
\begin{aligned} 
\boldsymbol{r}_j =&\, [r_{j, 1},r_{j, 2},\cdots,r_{j, d}]\\ 
\boldsymbol{\omega}_j =&\, [\omega_{j, 1},\omega_{j, 2},\cdots,\omega_{j, d}]\\ 
\boldsymbol{\theta}_j =&\, [\theta_{j, 1},\theta_{j, 2},\cdots,\theta_{j, d}]\\ 
\end{aligned}
\end{equation}
$$
 

代表词 $$j$$ 的三组词向量。你没看错，它确实假设每个词有三组跟位置无关的词向量了（当然可以按照某种形式进行参数共享，使得它退化为两组甚至一组），然后跟位置 $$k$$ 相关的词向量就按照上述公式运算。

你以为引入多组词向量就是它最特立独行的地方了？并不是！我们看到式 $$\eqref{eq:complex}$$ 还是复数形式，你猜它接下来怎么着？将它实数化？非也，它是将它直接用于复数模型！

也就是说，它走的是一条 **复数模型** 路线，不仅仅输入的 Embedding 层是复数的，里边的每一层 Transformer 都是复数的，它还实现和对比了复数版的 Fasttext、LSTM、CNN 等模型！

> 这篇文章的一作是 Benyou Wang，可以搜到他的相关工作基本上都是围绕着复数模型展开的，可谓复数模型的铁杆粉了～


#### 3.3、融合式：旋转位置编码（RoPE）






## 参考

- 苏剑林博客：

  - [让研究人员绞尽脑汁的Transformer位置编码](https://kexue.fm/archives/8130)

  - [层次分解位置编码，让BERT可以处理超长文本](https://kexue.fm/archives/7947)

  - [Transformer升级之路：2、博采众长的旋转式位置编码](https://kexue.fm/archives/8265)

- 知乎：[中文语言模型研究：(1) 乘性位置编码](https://zhuanlan.zhihu.com/p/183234823)


