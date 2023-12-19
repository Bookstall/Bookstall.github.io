---
layout: fragment
title: PyTorch：初始化模型参数
tags: [PyTorch]
excerpt: PyTorch：初始化模型参数
keywords: PyTorch
mathjax: true
---

PyTorch 在定义模型时有默认的参数初始化，有时候我们需要自定义参数的初始化，就需要用到 `torch.nn.init`。

## 模型参数初始化的两种方法

### 1）

- 先定义初始化模型方法；

- 再调用 `apply()`；

```python
class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), 
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
 
 # 1. 根据网络层的不同定义不同的初始化方式     
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为 conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# 2. 初始化网络结构        
model = Net(in_dim, n_hidden_1, n_hidden_2, out_dim)

# 3. 将 weight_init 应用在子模块上
model.apply(weight_init)
# torch 中的 apply 函数通过可以不断遍历 model 的各个模块。实际上其使用的是深度优先算法
```


### 2）

参数初始化定义在模型中，利用 `self.modules()` 来进行循环，如下所示：

```python
class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), 
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Linear(n_hidden_2, out_dim)
        )
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
```

> 注意：
> 
> 如果这个网络存在 backbone ，且其中有加载的预训练参数，在设置 `require_grad=False` 之后，初始化是否还会对 backbone 中网络起作用？
> 
> 这个要看你是先初始化还是先加载预训练参数了，**先初始化再加载预训练参数的话**，是 **不会改变** 预训练参数的


## 具体的初始化方法

PyTorch 中提供了 10 种初始化方法

- Xavier 均匀分布

- Xavier 正态分布

- Kaiming 均匀分布

- Kaiming 正态分布

- 均匀分布

- 正态分布

- 常数分布

- 正交矩阵初始化

- 单位矩阵初始化

- 稀疏矩阵初始化

在 `torch.nn.init` 中的各种初始化方法中，如 `nn.init.constant_(m.weight, 1)`，`nn.init.constant_(m.bias, 0)` 中第一个参数是 `tensor`，也就是对应的参数。


### Xavier



PyTorch 中 Linear 层使用的默认初始化方法是 Kaiming 初始化（也称为 He 初始化），其主要思想是根据该层输入特征的数量来初始化权重，从而减少梯度消失和梯度爆炸问题。Kaiming 初始化可以根据正态分布或均匀分布来初始化权重，具体分布类型取决于是否将非线性函数应用于神经网络的输出。在 PyTorch 中，可以通过将权重初始化方法设置为“kaiming_normal”或“kaiming_uniform”来实现 Kaiming 初始化。

具体来说，在 PyTorch 中，Linear 层的默认初始化方法是“kaiming_uniform”，其源代码如下：

```python
def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
```

可以看到，源代码中使用了 `kaiming_uniform_` 方法来初始化权重，并设置了参数 `a=math.sqrt(5)`。此外，如果存在偏置项，则使用均匀分布初始化偏置项，具体分布范围取决于输入特征数量。


## 参考

- 知乎：[pytorch初始化模型参数的两种方法](https://zhuanlan.zhihu.com/p/188701989)

- 


