---
layout: post
author: bookstall
tags: PyTorch
categories: [PyTorch]
excerpt: PyTorch 中的学习率衰减策略
keywords: PyTorch
title: 学习率衰减策略
mathjax: true
---

**学习率衰减（Learning Rate Decay）** 是一个非常有效的炼丹技巧，在神经网络的训练过程中，当 accuracy 出现震荡或 loss 不再下降时，进行适当的学习率衰减是一个行之有效的手段，很多时候能明显提高 accuracy。

几乎所有的神经网络采取的是梯度下降法来对模型进行最优化，其中标准的权重更新公式：

$$
W += \alpha * gradient
$$

- 学习率 $$\alpha$$ 控制着梯度更新的步长（step），$$\alpha$$ 越大，意味着下降的越快，到达最优点的速度也越快，如果为 $$0$$，则网络就会停止更新；

- 学习率过大，在算法优化的前期会加速学习，使得模型更容易接近局部或全局最优解。但是在后期会有较大波动，甚至出现损失函数的值围绕最小值徘徊，波动很大，始终难以达到最优。

所以引入学习率衰减的概念，直白点说，就是 <u>在模型训练初期，会使用较大的学习率进行模型优化，随着迭代次数增加，学习率会逐渐进行减小，保证模型在训练后期不会有太大的波动，从而更加接近最优解</u>。

PyTorch 中有两种学习率调整（衰减）的方法：

- 手动调整；

- 使用库函数进行调整；




## 手动调整学习率





## PyTorch 提供的学习率衰减策略


PyTorch 学习率调整策略通过 `torch.optim.lr_sheduler` 接口实现，`torch.optim.lr_scheduler` 提供了多种调整学习率的方法。

pytorch 提供的学习率调整策略分为三大类，分别是：

- 有序调整：等间隔调整（Step），多间隔调整（MultiStep），指数衰减（Exponential），余弦退火（CosineAnnealing）;

- 自适应调整：依训练状况伺机而变，通过监测某个指标的变化情况（loss、accuracy），当该指标不怎么变化时，就是调整学习率的时机（ReduceLROnPlateau）;

- 自定义调整：通过自定义关于 epoch 的 lambda 函数调整学习率（LambdaLR）。


通常使用的模板：

```python
scheduler = ...
for epoch in range(100):
    for data in dataloader:
        train(...)
        validate(...)
    scheduler.step()
```

###



### ReduceLROnPlateau

`torch.optim.lr_scheduler.ReduceLROnPlateau` 允许基于一些验证指标来动态地、自动地降低学习率。通常在发现 **loss 不再降低或者 acc 不再提高** 之后，降低学习率。

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.1, 
    patience=10, 
    verbose=False, 
    threshold=0.0001, 
    threshold_mode='rel', 
    cooldown=0,
    min_lr=0, 
    eps=1e-08
)
```

|      参数      |                                                                                                                    含义                                                                                                                    |
| :------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      mode      |                                                                                   'min' 模式检测 metric 是否不再减小，'max' 模式检测 metric 是否不再增大                                                                                   |
|     factor     |                                                                                                          触发条件后 lr *=f actor                                                                                                           |
|    patience    |                                                                                                        不再减小（或增大）的累计次数                                                                                                        |
|    verbose     |                                                                                                        触发条件后进行打印（print）                                                                                                         |
|   threshold    |                                                                                                          只关注超过阈值的显著变化                                                                                                          |
| threshold_mode | 有 rel 和 abs 两种阈值计算模式<br />rel 规则：max 模式下如果超过 best(1+threshold) 为显著，min 模式下如果低于 best(1-threshold) 为显著；<br />abs 规则：max 模式下如果超过 best+threshold 为显著，min 模式下如果低于 best-threshold 为显著 |
|    cooldown    |                                                                                        触发一次条件后，等待一定 epoch 再进行检测，避免 lr 下降过速                                                                                         |
|     min_lr     |                                                                                                               允许的最小 lr                                                                                                                |
|      eps       |                                                                                              如果新旧 lr 之间的差异小与 1e-8，则忽略此次更新                                                                                               |

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

scheduler.step()
```




## 打印学习率

```python
print("Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
```



## 参考

- 简书：[Pytorch中的学习率衰减及其用法](https://www.jianshu.com/p/26a7dbc15246) :star:

- 博客：[Pytorch使用ReduceLROnPlateau来更新学习率](https://www.emperinter.info/2020/08/05/change-leaning-rate-by-reducelronplateau-in-pytorch/)

- PyTorch Tutorial：[How to adjust learning rate](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

- CSDN：[pytorch 动态调整学习率，学习率自动下降，根据loss下降](https://blog.csdn.net/qq_41554005/article/details/119879911) :star:




