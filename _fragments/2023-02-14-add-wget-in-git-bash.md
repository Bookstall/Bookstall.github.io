---
layout: fragment
title: 在 git bash 窗口中添加 wget 命令（Windows）
tags: [git]
description: some word here
keywords: git
---

Windows 中 git bash 完全可以替代原生的 cmd，但是对于git bash会有一些Linux下广泛使用的命令的缺失，比如 `wget` 命令。

我们可能会遇到这种情况：在 windows 安装的 git bash 无法使用 `wget` 命令，如下图所示。

![无法使用 wget 命令](https://img-blog.csdnimg.cn/20200608154953297.png)

解决方法：

- 下载 wget 二进制安装包 [下载地址](https://eternallybored.org/misc/wget/)

- 解压安装包，将 `wget.exe` 拷贝到 `C:\Program Files\Git\mingw64\bin\` 下面；
  - 或者解压之后将解压文件中 `wget.exe` 的路径添加到环境变量中

经过上述步骤之后，即可在 windows git bash 中使用 `wget` 命令了，如下所示。

![可以使用 wget 命令](https://img-blog.csdnimg.cn/20200608155507101.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2VkZHkyMzUxMw==,size_16,color_FFFFFF,t_70)

## 参考

- CSDN：[windows git bash wget: command not found](https://blog.csdn.net/eddy23513/article/details/106621754)





