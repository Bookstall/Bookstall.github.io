---
layout: fragment
title: 离线迁移 Conda 环境
tags: [Conda]
excerpt: 离线迁移 Conda 环境
keywords: Conda
mathjax: true
---

## 迁出机器：打包环境

安装打包工具：

```shell
$ conda install -c conda-forge conda-pack
```

安装好之后打包需要迁出的环境：

```shell
$ conda pack -n envsname -o conda_envsname.tar.gz

Collecting packages...
Packing environment at '/opt/conda/envs/vim' to 'vim.tar.gz'
[########################################] | 100% Completed |  5min  5.6s
```

`-n` 之后为 虚拟环境名字，`-o` 之后为打包出来的文件名。

## 迁入机器：解压、部署环境

将打包的环境 `conda_envsname.tar.gz` 通过 ftp 传输到迁入机器中。

在你的 anaconda 目录下创建文件夹，名称（envs）即为你迁过来的环境名称：

```shell
$ mkdir -p /opt/conda/envs/envsname
```

解压环境：

```shell
$ tar -xzf /root/tempfile/conda_envsname.tar.gz -C /opt/conda/envs/envsname
```

`-C` 之前为打包压缩文件路径；`-C` 之后为迁入机器 conda 文件夹下 envs 目录 + 环境名。

执行后完成 `cd` 进 `envs` 目录中已经可以看到环境拷贝完成：

```shell
$ /opt/conda/envs/envsname
```

检查环境是否完全复制：

```shell
$ conda activate envsname

$ pip list

$ conda list
```

## 参考

- 知乎：[Conda 环境离线迁移（服务器断网情况下搭建虚拟环境 envs）](https://zhuanlan.zhihu.com/p/625457511)


