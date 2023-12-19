---
layout: fragment
title: PyTorch Error in rnn.pack_padded_sequence()
tags: [PyTorch]
description: some word here
keywords: PyTorch
---

## An PyTorch Error

在使用 GPU 训练模型时，PyTorch 代码中包含 `nn.utils.rnn.pack_padded_sequence()` 方法：

```python
pack_wemb = nn.utils.rnn.pack_padded_sequence(wemb, length, batch_first=True, enforce_sorted=False)
```

遇到了以下错误：

```shell
RuntimeError: 'lengths' argument should be a 1D CPU int64 tensor, but got 1D cuda:0 Long tensor
```

导致这个错误的原因是：PyTorch 1.6 及以后的版本对 Bi-LSTM 进行了升级/改动。因此，只有 PyTorch 1.5（包含）以前的版本能够正常使用。

## 解决方法

根据报错信息，我们最直接能想到的方法是 **将 PyTorch Tensor 移动到 CPU device 中**，即 `length.cpu()` 或者 `length.to("cpu")`：

```python
# Debug Issue: https://github.com/pytorch/pytorch/issues/43227
pack_wemb = nn.utils.rnn.pack_padded_sequence(wemb, length.cpu(), batch_first=True, enforce_sorted=False)
```

此外，还可以将 PyTorch 进行降级，从而解决这个问题（但是不推荐这样做）。


## 参考

- [PyTorch's issue 43227](https://github.com/pytorch/pytorch/issues/43227)

- [DeepCTR-Torch's issue 240](https://github.com/shenweichen/DeepCTR-Torch/issues/240)

- StackOverflow：[RuntimeError: ‘lengths’ argument should be a 1D CPU int64 tensor, but got 1D cuda:0 Long tensor](https://stackoverflow.com/questions/70428140/runtimeerror-lengths-argument-should-be-a-1d-cpu-int64-tensor-but-got-1d-cud)
