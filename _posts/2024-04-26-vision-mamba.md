---
layout: post
author: bookstall
tags: SSM
categories: [SSM]
excerpt: Vision Mamba
keywords: SSM
title: Vision Mamba
mathjax: true
---


##

![](https://img-blog.csdnimg.cn/direct/21cc44ac5624495f994f06222e9c2991.png)

![](https://img-blog.csdnimg.cn/direct/52e5cf69a2bf4bfb9942266de8d56507.png)


## 环境设置

- 操作系统：Ubuntu 18.04、CUDA 12.2、Python 3.10

Clone Vision Mamba 项目：

```shell
$ git clone https://github.com/hustvl/Vim.git
```

创建一个新的 Python 虚拟环境 `vim`：

```shell
$ conda create -n vim python=3.10.13
```

安装 Torch：

```shell
$ pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

$ pip list | grep torch
torch              2.1.1+cu121
torchaudio         2.1.1+cu121
torchvision        0.16.1+cu121
```

安装其他依赖：

```shell
$ pip install -r vim/vim_requirements.txt
```

安装 casual-conv1d：

```shell
$ pip install causal-conv1d==1.1.1

$ pip list | grep conv1d
causal-conv1d      1.1.1
```

由于网络的原因，这里使用本地安装 `mamba`：

首先去 [mamba](https://github.com/state-spaces/mamba/releases) 下载相应的文件 `mamba_ssm-1.1.1+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`，然后进行安装。`mamba` 的版本要与 `causal-conv1d` 的版本一致。

```shell
$ pip install mamba_ssm-1.1.1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 

$ pip list | grep mamba
mamba-ssm          1.1.1
```

由于 Vision Mamba 使用了双向的状态选择，作者重写了 `mamba-ssm `中的状态扫描方法。因此，我们需要将作者修改过的 `mamba-ssm` 覆盖原始的 `mamba-ssm`：

```shell
# 删除原始的 mamba-ssm
$ rm -rf /opt/conda/envs/vim/lib/python3.10/site-packages/mamba_ssm/

# 覆盖 vision mamba 的 mamba-ssm
$ cp -r mamba-1p1p1/mamba_ssm/ /opt/conda/envs/vim/lib/python3.10/site-packages/
```

最终，我们可以通过下面的代码，打印出 Vision Mamba Tiny 模型的结果，来验证环境是否正确：

```python
import torch
import torch.nn as nn
from timm.models import create_model
from pprint import pprint
import models_mamba

model = create_model(
    "vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2",
    pretrained=False,
    num_classes=1000,
    drop_rate=0.0,
    drop_path_rate=0.0,
    drop_block_rate=None,
    img_size=224
)

pprint(model)
```

输出结果为：

```shell
VisionMamba(
  (patch_embed): PatchEmbed(
    (proj): Conv2d(3, 192, kernel_size=(16, 16), stride=(16, 16))
    (norm): Identity()
  )
  (pos_drop): Dropout(p=0.0, inplace=False)
  (head): Linear(in_features=192, out_features=1000, bias=True)
  (drop_path): Identity()
  (layers): ModuleList(
    (0-23): 24 x Block(
      (mixer): Mamba(
        (in_proj): Linear(in_features=192, out_features=768, bias=False)
        (conv1d): Conv1d(384, 384, kernel_size=(4,), stride=(1,), padding=(3,), groups=384)
        (act): SiLU()
        (x_proj): Linear(in_features=384, out_features=44, bias=False)
        (dt_proj): Linear(in_features=12, out_features=384, bias=True)
        (conv1d_b): Conv1d(384, 384, kernel_size=(4,), stride=(1,), padding=(3,), groups=384)
        (x_proj_b): Linear(in_features=384, out_features=44, bias=False)
        (dt_proj_b): Linear(in_features=12, out_features=384, bias=True)
        (out_proj): Linear(in_features=384, out_features=192, bias=False)
      )
      (norm): RMSNorm()
      (drop_path): Identity()
    )
  )
  (norm_f): RMSNorm()
)
```



## 前向过程

### ViM block

![ViM block 的计算过程](https://img-blog.csdnimg.cn/direct/595835768d0a40eaaa7380208a1c97f7.png)


## 训练



## 测试





## 总结



## 参考

- CSDN：

  - [Vision Mamba 完美复现](https://blog.csdn.net/weixin_45743271/article/details/137753402)

  - [解决 timm 中自己 @register_model 注册模型时创建模型时找不到的问题](https://blog.csdn.net/weixin_47994925/article/details/129745845)

- Github：

  - Official Implementation：[Vim](https://github.com/hustvl/Vim)

  - Mini Implementation：[VisionMamba](https://github.com/kyegomez/VisionMamba)
    
    - 实现的 [代码](https://github.com/kyegomez/VisionMamba/blob/main/vision_mamba/model.py) 非常简洁，能够直观看到 Vision Mamba 的前向过程
    
    - 基于 [zetascale](https://github.com/kyegomez/zeta) 提供的 SSM 实现

    - `from zeta.nn import SSM`

  - [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/)

  - [mamba](https://github.com/state-spaces/mamba)

  - [mamba-minimal)](https://github.com/johnma2006/mamba-minimal)

