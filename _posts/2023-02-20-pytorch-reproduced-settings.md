---
layout: post
author: bookstall
tags: PyTorch
categories: [PyTorch]
description: PyTorch 中关于实验可复现的设置
keywords: PyTorch
title: PyTorch 中关于实验可复现的设置
mathjax: true
---

## 代码

在深度学习训练中，为什么使用同样的代码，与和作者/别人得到的结果不一样，或者每次跑同一个实验得到的结果都不一样。

上述问题主要涉及到 **随机数/随机种子** 的设定，当我们 **设置了具体的随机种子** 之后，每次实验所使用的随机数将是一致的，进而得到的实验结果也会保持一致。

> 可以将这段代码添加到 PyCharm 的 **活动模板（Live Template）** 中

```python
import numpy as np
import torch
import random
import os

seed_value = 2020   # 设定随机数种子

np.random.seed(seed_value) # Numpy 的随机种子
random.seed(seed_value) # random 库的随机种子
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止 hash 随机化，使得实验可复现

torch.manual_seed(seed_value)     # 为 CPU 设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前 GPU 设置随机种子（只用一块 GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有 GPU 设置随机种子（多块 GPU）

torch.backends.cudnn.deterministic = True
```

喜欢省流的读者可以就此打住。

下面对上述代码进行更一步的说明，主要包括三个方面：

- Python 和 Numpy 的随机数

- PyTorch 的随机数

- cuDNN 的随机数

### Python 和 Numpy 的随机数

如果读取数据的过程采用了 **随机预处理**（如 `RandomCrop`、`RandomHorizontalFlip` 等)，那么对 python、numpy 的随机数生成器也需要设置种子。

```python
np.random.seed(seed_value) # Numpy 的随机种子
random.seed(seed_value) # random 库的随机种子
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止 hash 随机化，使得实验可复现
```

关于 `os.environ['PYTHONHASHSEED']` 的详细描述，参见 Python 3.5 文档：[Environment variables：PYTHONHASHSEED](https://docs.python.org/3.5/using/cmdline.html#envvar-PYTHONHASHSEED)


### PyTorch 的随机数

在 PyTorch 中，会对 **模型的权重** 等进行初始化，因此也要设定随机数种子。

```python
torch.manual_seed(seed_value)     # 为 CPU 设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前 GPU 设置随机种子（只用一块 GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有 GPU 设置随机种子（多块 GPU）
```

另外，也有人提到说 **dataloder** 中，可能由于 **读取顺序不同，也会造成结果的差异**。

这主要是由于 dataloader 采用了 **多线程**，即 `num_workers > 1`。目前暂时没有发现解决这个问题的方法，但是只要 **固定 `num_workers` 数目（线程数）不变**，基本上也能够重复实验结果。



### cuDNN 的随机数

cuDNN 中对 **卷积操作** 进行了优化，牺牲了精度来换取计算效率。如果需要保证可重复性，可以使用如下设置：

```python
torch.backends.cudnn.deterministic = True
```



## 参考

- CSDN：[Pytorch中设置哪些随机数种子，才能保证实验可重复](https://blog.csdn.net/u014264373/article/details/114323297)

- Python 3.5 文档：[Environment variables：PYTHONHASHSEED](https://docs.python.org/3.5/using/cmdline.html#envvar-PYTHONHASHSEED)


